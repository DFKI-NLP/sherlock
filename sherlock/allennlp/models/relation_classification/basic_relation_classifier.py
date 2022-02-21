# -*- coding: utf8 -*-
"""

@date: 09.02.22
@source: https://github.com/DFKI-NLP/RelEx/blob/master/relex/models/relation_classification/basic_relation_classifier.py
@author: christoph.alt@posteo.de, gabriel.kressin@dfki.de
"""

from typing import Dict, Optional, List, Any, Set, Tuple
from collections import defaultdict

import torch
import numpy
from overrides import overrides

from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure


# from: https://github.com/DFKI-NLP/RelEx/blob/master/relex/modules/nn.py
class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters)
    in embedding space.
    """

    def __init__(self, p: float = 0.05, fill_idx: int = 1):
        super(WordDropout, self).__init__()
        self.prob = p
        self.fill_idx = fill_idx

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        if not self.training or not self.prob:
            return inputs

        dropout_mask = inputs.data.new(1, inputs.size(1)).bernoulli_(self.prob)
        dropout_mask = torch.autograd.Variable(dropout_mask, requires_grad=False)
        dropout_mask = dropout_mask * mask
        dropout_mask = dropout_mask.expand_as(inputs)
        return inputs.masked_fill_(dropout_mask.byte(), self.fill_idx)


@Model.register("basic_relation_classifier")
class BasicRelationClassifier(Model):
    """
    This ``Model`` performs relation classification on a given input.
    We assume we're given a text, head and tail entity offsets, and we predict the
    relation between head and tail entity. The basic model structure: we'll embed the
    text, relative head and tail offsets, and encode it with a Seq2VecEncoder, getting a
    single vector representing the content.  We'll then pass the result through a
    feedforward network, the output of which we'll use as our scores for each label.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    text_encoder : ``Seq2VecEncoder``, required
        The encoder that converts the input text to a vector.
    classifier_feedforward : ``FeedForward``, required
        The feedforward network that predicts the relation from the vector
        representation of the text.
    word_dropout : ``float```, optional (default=``0.``)
        Probability that a word/token is replaced by the OOV token.
    embedding_dropout : ``float``, optional (default=``0.``)
        Embedding dropout probability.
    encoding_dropout : ``float``, optional (default=``0.``)
        Text encoding dropout probability.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    offset_embedder_head : ``OffsetEmbedder``
        The embedder we use to embed each tokens relative offset to the head entity.
    offset_embedder_tail : ``OffsetEmbedder``
        The embedder we use to embed each tokens relative offset to the tail entity.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    verbose_metrics : ``bool``, optional (default=``False``)
        If true, output per-label metrics instead of aggregated precision, recall, and F1.
    ignore_label : ``str``, optional (default=``None``)
        If set, the label is ignored during metrics computation (e.g. for no_relation).
    f1_average : ``str```, optional (default=``macro``)
        Averaging method ("micro" or "macro") to compute the aggregated F1 score.
    use_adjacency : ``bool``
        If true, the adjacency matrix computed from the dependency parse is passed to the
        text encoder.
    use_entity_offsets : ``bool``
        If true, head and tail spans are passed to the text encoder.
    weights: ``torch.Tensor``, optional (default = None)
        Weights for class labels. Useful for unbalanced training sets.
    label_namespace : ``str``, optional (default=``"labels"``)
        Gives Namespace for labels within vocabulary.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        text_encoder: Seq2VecEncoder,
        classifier_feedforward: FeedForward,
        word_dropout: float = 0.,
        embedding_dropout: float = 0.,
        encoding_dropout: float = 0.,
        #  offset_embedder_head: Optional[OffsetEmbedder] = None,
        #  offset_embedder_tail: Optional[OffsetEmbedder] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        verbose_metrics: bool = False,
        ignore_label: str = None,
        f1_average: str = "macro",
        use_adjacency: bool = False,
        use_entity_offsets: bool = False,
        weights: torch.Tensor = None,
        label_namespace: str = "labels",
    ) -> None:
        super(BasicRelationClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        self.text_encoder = text_encoder

        self.word_dropout = (
                WordDropout(word_dropout, fill_idx=vocab.get_token_index(DEFAULT_OOV_TOKEN))
                if word_dropout > 0. else lambda x, m: x)
        self.embedding_dropout = (
                torch.nn.Dropout(encoding_dropout)
                if embedding_dropout > 0. else lambda x: x)
        self.encoding_dropout = (
                torch.nn.Dropout(encoding_dropout)
                if encoding_dropout > 0. else lambda x: x)

        self.classifier_feedforward = classifier_feedforward
        # self.offset_embedder_head = offset_embedder_head
        # self.offset_embedder_tail = offset_embedder_tail
        self._verbose_metrics = verbose_metrics
        self._use_adjacency = use_adjacency
        # Loading previously trained GCNs requires entity
        # offsets as well as the adjacency matrix
        self._use_entity_offsets = use_entity_offsets or self._use_adjacency

        offset_embedding_dim = 0
        # if offset_embedder_head is not None:
        #     if not offset_embedder_head.is_additive():
        #         offset_embedding_dim += offset_embedder_head.get_output_dim()

        # if offset_embedder_tail is not None:
        #     if not offset_embedder_tail.is_additive():
        #         offset_embedding_dim += offset_embedder_tail.get_output_dim()

        text_encoder_input_dim = (text_field_embedder.get_output_dim()
                                  + offset_embedding_dim)

        if text_encoder_input_dim != text_encoder.get_input_dim():
            raise ConfigurationError(
                    "The output dimension of the text_field_embedder and offset_embedders "
                    "must match the input dimension of the text_encoder. Found {} and {}, "
                    "respectively.".format(text_field_embedder.get_output_dim(),
                                           text_encoder.get_input_dim()))

        if text_encoder.get_output_dim() != classifier_feedforward.get_input_dim():
            raise ConfigurationError(
                    "The output dimension of the text_encoder must match the "
                    "input dimension of the classifier_feedforward. Found {} and {}, "
                    "respectively.".format(text_encoder.get_output_dim(),
                                           classifier_feedforward.get_input_dim()))

        # if classifier_feedforward.get_output_dim() != self.num_classes:
        #     raise ConfigurationError(
        #             "The output dimension of the classifier_feedforward must match the "
        #             "number of classes in the dataset. Found {} and {}, "
        #             "respectively.".format(classifier_feedforward.get_output_dim(),
        #                                    self.num_classes))

        self.metrics = {"accuracy": CategoricalAccuracy()}
        self.f1_average = f1_average
        self.label_tokens = vocab.get_index_to_token_vocabulary(label_namespace)
        self.f1 = FBetaMeasure(
            average=f1_average,
            labels=[
                label_id
                for label_id, label in self.label_tokens.items()
                if label != ignore_label
            ],
        )

        self.loss = torch.nn.CrossEntropyLoss(weights)

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        text: Dict[str, torch.LongTensor],
        # head: torch.LongTensor,
        # tail: torch.LongTensor,
        adjacency: torch.LongTensor = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        label: Optional[torch.LongTensor] = None
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ,unused-argument
        """
        Parameters
        ----------
        text : Dict[str, torch.Tensor], required
            The output of ``TextField.as_array()``.
        head : torch.LongTensor,
        tail : torch.LongTensor,
        adjacency: torch.LongTensor
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        text_mask = util.get_text_field_mask(text)

        # TODO: make this generic
        if "tokens" in text:
            text["tokens"]["tokens"] = self.word_dropout(text["tokens"]["tokens"], text_mask)  # type: ignore

        embedded_text = self.text_field_embedder(text)

        embedded_text = self.embedding_dropout(embedded_text)

        embeddings = [embedded_text]
        # if self.offset_embedder_head is not None:
        #     embeddings.append(self.offset_embedder_head(embedded_text,
        #                                                 text_mask,
        #                                                 span=head))

        # if self.offset_embedder_tail is not None:
        #     embeddings.append(self.offset_embedder_tail(embedded_text,
        #                                                 text_mask,
        #                                                 span=tail))

        if len(embeddings) > 1:
            embedded_text = torch.cat(embeddings, dim=-1)
        else:
            embedded_text = embeddings[0]

        additional_encoder_args = {}
        # if self._use_entity_offsets:
        #     additional_encoder_args['head'] = head
        #     additional_encoder_args['tail'] = tail
        if self._use_adjacency:
            additional_encoder_args['adjacency'] = adjacency

        encoded_text = self.text_encoder(embedded_text,
                                         text_mask,
                                         **additional_encoder_args)

        encoded_text = self.encoding_dropout(encoded_text)

        logits = self.classifier_feedforward(encoded_text)

        output_dict = {"logits": logits, "input_rep": encoded_text}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            self.f1(logits.detach().cpu(), label.cpu())
            output_dict["loss"] = loss

        return output_dict


    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = torch.nn.functional.softmax(output_dict["logits"], dim=-1)
        output_dict["class_probabilities"] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict["label"] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset)
                             for metric_name, metric in self.metrics.items()}

        f1_dict = self.f1.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update(f1_dict)
        else:
            metrics_to_return.update({x: y for x, y in f1_dict.items()
                                      if "fscore" in x})

        return metrics_to_return