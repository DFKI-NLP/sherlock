# from here: https://github.com/DFKI-NLP/RelEx/blob/master/relex/models/relation_classification/basic_relation_classifier.py

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
from allennlp.training.metrics import CategoricalAccuracy, Metric

# from relex.modules.offset_embedders import OffsetEmbedder
# from relex.modules.nn import WordDropout
# from relex.metrics import F1Measure

# TODO: THIS IS TEMPORARY.
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


# TODO: THIS IS TEMPORARY.
# from: https://github.com/DFKI-NLP/RelEx/blob/master/relex/metrics/f1_measure.py
class F1Measure(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 average: str = "macro",
                 label_namespace: str = "labels",
                 ignore_label: str = None) -> None:
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(label_namespace)
        self._average = average
        self._ignore_label = ignore_label
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._true_negatives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        # TODO: debug
        # predictions, gold_labels, mask = self.unwrap_to_tensors(predictions,
        #                                                         gold_labels,
        #                                                         mask)

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            msg = ("A gold label passed to F1Measure contains an "
                   "id >= {}, the number of classes.".format(num_classes))
            raise ConfigurationError(msg)

        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.float()
        gold_labels = gold_labels.float()

        argmax_predictions = predictions.max(-1)[1].float().squeeze(-1)

        for label_index, label_token in self._label_vocabulary.items():

            positive_label_mask = gold_labels.eq(label_index).float()
            negative_label_mask = 1.0 - positive_label_mask

            # True Negatives: correct non-positive predictions.
            correct_null_predictions = (argmax_predictions != label_index).float() * negative_label_mask
            self._true_negatives[label_token] += (correct_null_predictions.float() * mask).sum()

            # True Positives: correct positively labeled predictions.
            correct_non_null_predictions = (argmax_predictions == label_index).float() * positive_label_mask
            self._true_positives[label_token] += (correct_non_null_predictions * mask).sum()

            # False Negatives: incorrect negatively labeled predictions.
            incorrect_null_predictions = (argmax_predictions != label_index).float() * positive_label_mask
            self._false_negatives[label_token] += (incorrect_null_predictions * mask).sum()

            # False Positives: incorrect positively labeled predictions
            incorrect_non_null_predictions = (argmax_predictions == label_index).float() * negative_label_mask
            self._false_positives[label_token] += (incorrect_non_null_predictions * mask).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}

        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(
                    self._true_positives[tag],
                    self._false_positives[tag],
                    self._false_negatives[tag],
            )
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        if self._average == "micro":
            if self._ignore_label is not None:
                precision, recall, f1_measure = self._compute_metrics(
                        sum([val for l, val in self._true_positives.items()
                             if l != self._ignore_label]),
                        sum([val for l, val in self._false_positives.items()
                             if l != self._ignore_label]),
                        sum([val for l, val in self._false_negatives.items()
                             if l != self._ignore_label]))
            else:
                precision, recall, f1_measure = self._compute_metrics(
                        sum(self._true_positives.values()),
                        sum(self._false_positives.values()),
                        sum(self._false_negatives.values()),
                )
        elif self._average == "macro":
            precision = 0.0
            recall = 0.0
            n_precision = 0
            n_recall = 0

            for tag in all_tags:
                precision_key = "precision" + "-" + tag
                recall_key = "recall" + "-" + tag
                precision += all_metrics[precision_key]
                recall += all_metrics[recall_key]
                n_precision += 1
                n_recall += 1

            if n_precision:
                precision /= n_precision
            if n_recall:
                recall /= n_recall
            f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))

        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int,
                         false_positives: int,
                         false_negatives: int) -> Tuple[float, float, float]:
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._true_negatives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)


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
    """

    def __init__(self,
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
    ) -> None:
        super(BasicRelationClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
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
        self._f1_measure = F1Measure(vocabulary=self.vocab,
                                     average=f1_average,
                                     ignore_label=ignore_label)

        self.loss = torch.nn.CrossEntropyLoss(weights)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                # head: torch.LongTensor,
                # tail: torch.LongTensor,
                adjacency: torch.LongTensor = None,
                metadata: Optional[List[Dict[str, Any]]] = None,
                label: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
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
            text["tokens"] = self.word_dropout(text["tokens"], text_mask)  # type: ignore

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
            self._f1_measure(logits, label)
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

        f1_dict = self._f1_measure.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update(f1_dict)
        else:
            metrics_to_return.update({x: y for x, y in f1_dict.items()
                                      if "overall" in x})

        return metrics_to_return