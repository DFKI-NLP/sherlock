# -*- coding: utf8 -*-
"""

@date: 11.01.2022
@author: gabriel.kressin@dfki.de
"""

import logging
from typing import Dict, Optional, Any

import torch

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure


logger = logging.getLogger(__name__)


@Model.register("transformer_relation_classificatier")
class TransformerRelationClassifier(Model):
    """
    This ``Model`` performs relation classification on a given input.

    It embeds the tokens with a Transformer, takes the embedding of the
    [CLS] token and classifies based on that onto a relation.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
        Needs ``"tokens"`` and ``label_namespace`` as namespaces.
    model_name : ``str``, optional (default=``"bert-base-uncased"``)
        Chooses underlying Transformer architecture based on this parameter.
        Has to be an available Huggingface transformer.
    max_length : ``int``, optional (default=``None``)
        If set to a number, will limit sequences to maximum length.
    override_weights_file: `Optional[str]`, optional (default = `None`)
        If set, this specifies a file from which to load alternate weights
        that override the weights from huggingface. The file is expected
        to contain a PyTorch `state_dict`, created with `torch.save()`.
    num_labels : ``int``, optional (default=``None``)
        Determines output dimension of Model. If not given, will be determined
        from `vocab`.
    label_namespace : ``str``, optional (default=``"labels"``)
        Gives Namespace for labels within vocabulary.
    pooler_dropout: ``float``, optional (default=``0.1``)
        Pooler dropout probability.
    ignore_label : ``str``, optional (default=``None``)
        If set, the label is ignored during metrics computation
        (e.g. for no_relation).
    f1_average : ``str```, optional (default=``macro``)
        Averaging method ("micro", "macro", "weighted" or "none") to compute
        the aggregated F1 score.
    weights : ``torch.Tensor``, optional (default = ``None``)
        Weights for class labels. Useful for unbalanced training sets.
    tokenizer_kwargs : ``Dict``, optional (default = ``None``)
        Arguments to initialize a tokenizer to potentially resize
        embedding layer for the Transformer. This is the only way allennlp
        allows you to do that.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str = "bert-base-uncased",
        max_length: Optional[int] = None,
        override_weights_file: Optional[str] = None,
        num_labels: Optional[int] = None,
        label_namespace: str = "labels",
        pooler_dropout: float = 0.1,
        ignore_label : Optional[str] = None,
        f1_average: str = "macro",
        weights: torch.Tensor = None,
        tokenizer_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.embedder = BasicTextFieldEmbedder(
            {"tokens": PretrainedTransformerEmbedder(
                model_name=model_name,
                max_length=max_length,
                override_weights_file=override_weights_file,
                tokenizer_kwargs=tokenizer_kwargs,
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

        self.loss = torch.nn.CrossEntropyLoss(weights)
        self.acc = CategoricalAccuracy()
        self.f1_average = f1_average
        self.f1 = FBetaMeasure(
            average=f1_average,
            labels=[
                label_id
                for label_id, label in self.label_tokens.items()
                if label != ignore_label
            ],
        )


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
            A scalar loss to be optimised. This is only returned when `label`
            is not `None`.
        logits : ``torch.FloatTensor``
            The logits for every possible answer choice
        """

        embedded = self.embedder(text)

        pooled = self.pooler(embedded)

        logits = self.fc_out(pooled)

        result = {"logits": logits}

        if label is not None:
            result["loss"] = self.loss(logits, label)
            self.acc(logits.detach(), label)
            self.f1(logits.detach().cpu(), label.cpu())

        return result


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = {"acc": self.acc.get_metric(reset)}
        for metric_name, metric_value in self.f1.get_metric(reset).items():
            if self.f1_average == "none":
                for class_index, value in enumerate(metric_value):
                    result[f"{self.label_tokens[class_index]}-{metric_name}"] = value
            else:
                result[metric_name] = metric_value
        return result
