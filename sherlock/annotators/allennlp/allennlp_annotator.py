#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 08.12.21
@author: leonhard.hennig@dfki.de
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import json
from torch.utils.data import DataLoader, SequentialSampler

from sherlock import Document
from sherlock.dataset import InstancesDataset
from sherlock.feature_converters import FeatureConverter
from sherlock.annotators.annotator import Annotator
from sherlock.tasks import NLP_TASK_CLASSES, NLPTask

from allennlp.data.tokenizers import Tokenizer, PreTrainedTransformerTokenizer
from allennlp.models.model import Model
from allennlp.models.archival import load_archive
from allennlp.common.file_utils import cached_path


class AllenNLPAnnotator(Annotator):
    name = ""
    task = NLPTask.NONE

    def __init__(
        self,
        tokenizer: Tokenizer,
        converter: FeatureConverter,
        model: Model, # fix
        device: str = "cpu",
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.converter = converter
        self.model = model.to(device) # taken from transformers_annotator.py, but not sure if this actually works acc. to doc of torch.nn.Module (https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(  # type: ignore
            cls, path: str, **kwargs
    ) -> "Annotator":
        #_, model_class, tokenizer_class = NLP_TASK_CLASSES[cls.task][args.model_type]
        #archive = load_archive(archive_file=cached_path(path),
        #                       cuda_device=kwargs.get("device", "cpu"),
        #                       overrides=kwargs.get("config_overrides", json.dumps({})))
        #model = model_class.from_archive(archive=archive, predictor_name=predictor_name)
        pass # todo

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

    # todo test. this method is currently just copied from transformers_annotator.py, with the minor modification
    # of reading "Instance"s from the inputfeatures as input to the dataloader
    def _convert_and_annotate(
            self, documents: List[Document]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict[str, Any]]]:
        input_features = self.converter.documents_to_features(documents)
        instances = [f.instance for f in input_features]

        eval_dataset = InstancesDataset(instances)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.batch_size)

        annot_list = []
        label_ids_list = []
        for batch in eval_dataloader:
            outputs = self.model.forward_on_instances(batch)

            #self.model.eval()
            #batch = {k: t.to(self.device) for k, t in batch.items()}
            #if self.unused_token_type_ids:
            #    batch["token_type_ids"] = None
            #if self.no_token_type_ids:
            #    del batch["token_type_ids"]

            #with torch.no_grad():
            #    outputs = self.model(**batch) # allennlp way: self.model.forward_on_instances()

            # todo: not sure if this works with forward_on_instances(). probably not
            logits = outputs[1] if "labels" in batch else outputs[0]
            annot_list.append(logits.detach().cpu().numpy())
            if "labels" in batch:
                label_ids_list.append(batch["labels"].detach().cpu().numpy())

        annotations = np.concatenate(annot_list, axis=0) if len(annot_list) > 0 else None
        label_ids = np.concatenate(label_ids_list, axis=0) if len(label_ids_list) > 0 else None
        return annotations, label_ids, [f.metadata for f in input_features]