#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 11.01.2022
@author: gabriel.kressin@dfki.de
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.transformer import TransformerEmbeddings, TransformerStack, TransformerPooler
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure


logger = logging.getLogger(__name__)


@Model.register("transformer_relation_classificatier")
class TransformerRelationClassifier(Model):
    """
    This model implements a Relation Classifier based on a Transformer
    to create embeddings for the inputs.

    Parameters
    ----------
    vocab : ``Vocabulary``
    transformer_model : ``str``, optional (default=``"bert-base-uncased"``)
        Chooses underlying Transformer based on this parameter.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str = "bert-base-uncased",
        max_length: Optional[int] = None,
        num_labels: Optional[int] = None,
        label_namespace: str = "labels",
        override_weights_file: Optional[str] = None,
        pooler_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.embedder = BasicTextFieldEmbedder(
            {"tokens": PretrainedTransformerEmbedder(
                model_name=model_name,
                max_length=max_length,
                override_weights_file=override_weights_file,
            )}
        )

        self.pooler = BertPooler(
            pretrained_model=model_name,
            dropout=pooler_dropout,
        )

        self.label_tokens = vocab.get_index_to_token_vocabulary(label_namespace)
        if num_labels is None:
            num_labels = len(self.label_tokens)
        self.fc_out = torch.nn.Linear(self.pooler.get_output_dim(), num_labels)
        self.fc_out.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_out.bias.data.zero_()

        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = CategoricalAccuracy()
        self.f1 = FBetaMeasure()


    def forward(  # type: ignore
        self,
        text: Dict[str, torch.Tensor],
        label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text : ``Dict[str, torch.LongTensor]``
            From a ``TensorTextField``. Contains the text to be classified.
        label : ``Optional[torch.LongTensor]``
            From a ``LabelField``, specifies the true class of the instance

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. This is only returned when `correct_alternative` is not `None`.
        logits : ``torch.FloatTensor``
            The logits for every possible answer choice
        """
        embedded = self.embedder(text)

        pooled = self.pooler(embedded)

        logits = F.softmax(self.fc_out(pooled), dim=-1)

        result = {"logits": logits}

        if label is not None:
            result["loss"] = self.loss(logits, label)
            self.acc(logits, label)
            self.f1(logits, label)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = {"acc": self.acc.get_metric(reset)}
        for metric_name, metrics_per_class in self.f1.get_metric(reset).items():
            for class_index, value in enumerate(metrics_per_class):
                result[f"{self.label_tokens[class_index]}-{metric_name}"] = value
        return result
