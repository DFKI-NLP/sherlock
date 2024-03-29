# -*- coding: utf8 -*-
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    BertPreTrainedModel,
    DistilBertPreTrainedModel,
    PreTrainedModel,
    XLNetPreTrainedModel,
)

from sherlock import Document
from sherlock.dataset import TensorDictDataset
from sherlock.feature_converters import FeatureConverter
from sherlock.annotators.annotator import Annotator
from sherlock.tasks import NLP_TASK_CLASSES, NLPTask


class TransformersAnnotator(Annotator):
    name = ""
    task = NLPTask.NONE

    def __init__(
        self,
        converter: FeatureConverter,
        model: PreTrainedModel,
        device: str = "cpu",
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        self.converter = converter
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

        self.unused_token_type_ids = isinstance(model, (BertPreTrainedModel, XLNetPreTrainedModel))
        self.no_token_type_ids = isinstance(model, (DistilBertPreTrainedModel))

    @classmethod
    def from_pretrained(  # type: ignore
        cls, path: str, **kwargs
    ) -> "Annotator":
        # TODO: not very consistent, some of the stuff comes from args, some from kwargs
        args = torch.load(os.path.join(path, "training_args.bin"))
        _, model_class, tokenizer_class = NLP_TASK_CLASSES[cls.task][args.model_type]
        tokenizer = tokenizer_class.from_pretrained(path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(path)
        converter = FeatureConverter.from_pretrained(path, tokenizer=tokenizer)
        return cls(
            converter,
            model,
            **{k: v for k, v in kwargs.items() if k in ["device", "batch_size", "ignore_no_relation", "add_logits"]},
        )

    def process_documents(self, documents: List[Document]) -> List[Document]:
        results = []  # type: List[Document]
        for i in range(0, len(documents), self.batch_size):
            batch_documents = documents[i : i + self.batch_size]
            annotations, label_ids, metadata = self._convert_and_annotate(batch_documents)
            results.extend(self.combine(documents, annotations, label_ids, metadata))
        return results

    def combine(
        self,
        documents: List[Document],
        annotations: Optional[np.ndarray],
        label_ids: Optional[np.ndarray],
        metadata: List[Dict[str, Any]],
    ) -> List[Document]:
        raise NotImplementedError("Annotator must implement 'combine'.")

    def _convert_and_annotate(
        self, documents: List[Document]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict[str, Any]]]:
        input_features = self.converter.documents_to_features(documents)

        tensor_dicts = []
        for features in input_features:
            # Fix to support RoBERTa models that do not make use of token type ids and may return None
            token_type_ids = features.token_type_ids if features.token_type_ids else [0 for _ in features.input_ids]
            tensor_dict = {
                "input_ids": torch.tensor(features.input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(features.attention_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            }
            if features.labels is not None:
                tensor_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
            tensor_dicts.append(tensor_dict)

        eval_dataset = TensorDictDataset(tensor_dicts)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.batch_size)

        annot_list = []
        label_ids_list = []
        for batch in eval_dataloader:
            self.model.eval()
            batch = {k: t.to(self.device) for k, t in batch.items()}
            if self.unused_token_type_ids:
                batch["token_type_ids"] = None
            if self.no_token_type_ids:
                del batch["token_type_ids"]

            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs[1] if "labels" in batch else outputs[0]
            annot_list.append(logits.detach().cpu().numpy())
            if "labels" in batch:
                label_ids_list.append(batch["labels"].detach().cpu().numpy())

        annotations = np.concatenate(annot_list, axis=0) if len(annot_list) > 0 else None
        label_ids = np.concatenate(label_ids_list, axis=0) if len(label_ids_list) > 0 else None
        return annotations, label_ids, [f.metadata for f in input_features]
