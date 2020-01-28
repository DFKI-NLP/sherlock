import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    BertPreTrainedModel,
    DistilBertPreTrainedModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    XLNetPreTrainedModel,
)

from sherlock import Document
from sherlock.dataset import TensorDictDataset
from sherlock.feature_converters import FeatureConverter
from sherlock.predictors import Predictor
from sherlock.tasks import NLP_TASK_CLASSES, NLPTask


class TransformersPredictor(Predictor):
    name = ""
    task = NLPTask.NONE

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        converter: FeatureConverter,
        model: PreTrainedModel,
        device: str = "cpu",
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.converter = converter
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

        self.unused_token_type_ids = isinstance(model, (BertPreTrainedModel, XLNetPreTrainedModel))
        self.no_token_type_ids = isinstance(model, (DistilBertPreTrainedModel))

    @classmethod
    def from_pretrained(  # type: ignore
        cls, path: str, device: str = "cpu", batch_size: int = 16, **kwargs
    ) -> "Predictor":
        args = torch.load(os.path.join(path, "training_args.bin"))
        _, model_class, tokenizer_class = NLP_TASK_CLASSES[cls.task][args.model_type]
        tokenizer = tokenizer_class.from_pretrained(path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(path)
        converter = FeatureConverter.from_pretrained(path, tokenizer)
        return cls(tokenizer, converter, model, device, batch_size, **kwargs)

    def predict_documents(self, documents: List[Document]) -> List[Document]:
        results = []  # type: List[Document]
        for i in range(0, len(documents), self.batch_size):
            batch_documents = documents[i : i + self.batch_size]
            predictions, label_ids, metadata = self._convert_and_predict(batch_documents)
            results.extend(self.combine(documents, predictions, label_ids, metadata))
        return results

    def combine(
        self,
        documents: List[Document],
        predictions: np.array,
        label_ids: np.array,
        metadata: List[Dict[str, Any]],
    ) -> List[Document]:
        raise NotImplementedError("Predictor must implement 'combine'.")

    def _convert_and_predict(
        self, documents: List[Document]
    ) -> Tuple[np.array, np.array, List[Dict[str, Any]]]:
        input_features = self.converter.documents_to_features(documents)

        tensor_dicts = []
        for features in input_features:
            tensor_dict = {
                "input_ids": torch.tensor(features.input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(features.attention_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(features.token_type_ids, dtype=torch.long),
            }
            if features.labels is not None:
                tensor_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
            tensor_dicts.append(tensor_dict)

        eval_dataset = TensorDictDataset(tensor_dicts)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.batch_size)

        predictions = None
        label_ids = None
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
            logits = torch.nn.functional.softmax(logits, dim=-1)
            if predictions is None:
                predictions = logits.detach().cpu().numpy()
                if "labels" in batch:
                    label_ids = batch["labels"].detach().cpu().numpy()
            else:
                predictions = np.append(predictions, logits.detach().cpu().numpy(), axis=0)
                if "labels" in batch:
                    label_ids = np.append(label_ids, batch["labels"].detach().cpu().numpy(), axis=0)
        return predictions, label_ids, [f.metadata for f in input_features]
