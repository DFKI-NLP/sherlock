# -*- coding: utf8 -*-
"""

@date: 09.02.22
@author: leonhard.hennig@dfki.de, gabriel.kressin@dfki.de
"""
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader, DataLoader
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn.util import move_to_device

from sherlock import Document
from sherlock.annotators.annotator import Annotator
from sherlock.feature_converters import FeatureConverter
from sherlock.tasks import NLP_TASK_CLASSES, NLPTask



class AllenNLPAnnotator(Annotator):
    name = ""
    task = NLPTask.NONE

    def __init__(
        self,
        converter: FeatureConverter,
        model: Model, # fix
        vocabulary: Vocabulary,
        device: str="cpu",
        batch_size: int=16,
        **kwargs,
    ) -> None:
        self.converter = converter
        self.model = model.to(device) # taken from transformers_annotator.py, but not sure if this actually works acc. to doc of torch.nn.Module (https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
        self.vocabulary = vocabulary
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(  # type: ignore
            cls, archive_file: str, cuda_device: int=-1, batch_size: int=16,
    ) -> "Annotator":
        """Returns an AllenNLPAnnotator. Expects as input an archival file or
        a directory containing an archival file in the format
        in which allennlp archives models, vocabularies and configs.
        http://docs.allennlp.org/main/api/models/archival/
        If trained with `allennlp train` this can just be the `serialization_dir`.

        """
        # Determine correct archive_file path
        if os.path.isdir(archive_file):
            # try default archive name
            archive_file = os.path.join(archive_file, "model.tar.gz")
        if not os.path.exists(archive_file):
            raise ConfigurationError(
                f"Archive file {archive_file} neither exists as file or dir."
            )

        archive = load_archive(
            archive_file=archive_file,
            cuda_device=cuda_device,
        )

        return cls(
            converter=archive.dataset_reader.feature_converter,
            model=archive.model,
            vocabulary=archive.model.vocab,
            device=cuda_device,
            batch_size=batch_size,
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

    # TODO: test.
    def _convert_and_annotate(
            self, documents: List[Document]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict[str, Any]]]:
        input_features = self.converter.documents_to_features(documents)
        instances = [f.instance for f in input_features]

        # eval_dataset = InstancesDataset(instances)
        # eval_sampler = SequentialSampler(eval_dataset)
        # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.batch_size)
        eval_dataloader = SimpleDataLoader(instances, self.batch_size)

        eval_dataloader.index_with(self.vocabulary)

        annot_list = []
        label_ids_list = []
        for batch in eval_dataloader:
            self.model.eval()

            batch = move_to_device(batch, self.device)

            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs["probs"]
            annot_list.append(logits.detach().cpu().numpy())
            if "labels" in batch:
                label_ids_list.append(batch["labels"].detach().cpu().numpy())

        annotations = np.concatenate(annot_list, axis=0) if len(annot_list) > 0 else None
        label_ids = np.concatenate(label_ids_list, axis=0) if len(label_ids_list) > 0 else None
        return annotations, label_ids, [f.metadata for f in input_features]
