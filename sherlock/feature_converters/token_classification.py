import logging
from typing import List

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers.token_class import Token
from torch.nn import CrossEntropyLoss
from transformers import XLNetTokenizer

from sherlock import Document
from sherlock.feature_converters.feature_converter import FeatureConverter, InputFeatures
from sherlock.feature_converters.input_features import InputFeaturesAllennlp, InputFeaturesTransformers



logger = logging.getLogger(__name__)

# TODO: this all needs testing.
# TODO: needs own pytest.


@FeatureConverter.register("token_classification")
class TokenClassificationConverter(FeatureConverter):
    """
    Class to convert Documents into InputFeatures used for Annotators
    in TokenClassification.

    Parameters
    ----------
    labels
    max_length
    framework
    entity_handling
    pad_token_segmend_id    # TODO: probably move into `transformers` kwargs
    log_num_input_features
    kwargs : ``Dict[str, any]``
        framwork specific keywords.
        `transformers`:
            tokenizer  : `PreTrainedTokenizer`
        `allennlp`:
            tokenizer : ``Tokenizer``
            token_indexers : ``Dict[str, TokenIndexer]``
    """

    def __init__(
        self,
        labels: List[str],
        max_length: int=512,
        framework: str="transformers",
        pad_token_segment_id: int=0,
        pad_token_label_id: int=CrossEntropyLoss().ignore_index,
        log_num_input_features: int=-1,
        **kwargs,
    ) -> None:
        super().__init__(labels, max_length, framework, **kwargs)
        self.pad_token_segment_id = pad_token_segment_id
        self.pad_token_label_id = pad_token_label_id
        self.log_num_input_features = log_num_input_features


    @property
    def name(self) -> str:
        return "token_classification"


    @property
    def persist_attributes(self) -> List[str]:
        return ["max_length", "pad_token_segment_id", "pad_token_label_id"]


    def document_to_features_transformers(
        self, document: Document, verbose: bool = False
    ) -> List[InputFeaturesTransformers]:
        tokens: List[str] = []
        labels: List[str] = []
        label_ids: List[int] = []
        for token in document.tokens:
            subword_tokens = self.tokenizer.tokenize(token.text)
            if len(subword_tokens) == 0:
                continue  # Skip whitespace tokens
            tokens.extend(subword_tokens)
            label = token.ent_type
            if label is None:
                label = "O"
            labels.append(label)
            # Use the real label id for the first token of the word,
            # and padding ids for the remaining tokens
            label_ids.extend(
                [self.label_to_id_map[label]]
                + [self.pad_token_label_id] * (len(subword_tokens) - 1)
            )

        inputs = self.tokenizer.encode_plus(
            text=tokens,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_overflowing_tokens=True,
        )

        num_truncated_tokens = inputs.get("num_truncated_tokens", 0)
        if num_truncated_tokens > 0:
            label_ids = label_ids[:-num_truncated_tokens]

        cls_token_at_end = isinstance(self.tokenizer, XLNetTokenizer)
        if cls_token_at_end:
            label_ids += [self.pad_token_label_id]
        else:
            label_ids = [self.pad_token_label_id] + label_ids

        padding_length = self.max_length - len(label_ids)
        if self.tokenizer.padding_side == "left":
            label_ids = ([self.pad_token_label_id] * padding_length) + label_ids
        else:
            label_ids += [self.pad_token_label_id] * padding_length

        metadata = dict(guid=document.guid, truncated=num_truncated_tokens > 0)

        features = InputFeaturesTransformers(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            labels=label_ids,
            metadata=metadata,
        )

        if verbose:
            self._log_input_features(tokens, document, features, labels)

        return [features]


    def document_to_features_allennlp(
        self, document: Document, verbose: bool = False
    ) -> List[InputFeaturesAllennlp]:
        # This implementation is kind of whacky and still needs
        # testing and bugfixing.
        tokens: List[Token] = []
        labels: List[str] = []
        label_ids: List[int] = []
        # In this case one document is treated as one entire
        # sequence for the model input
        for token in document.tokens:
            subword_tokens: List[Token] = self.tokenizer.tokenize(token.text)
            if len(subword_tokens) == 0:
                continue  # Skip whitespace tokens

            tokens.extend(subword_tokens)
            label = token.ent_type
            if label is None:
                label = "O"
            labels.append(label)
            # Use the real label id for the first token of the word,
            # and padding ids for the remaining tokens
            label_ids.extend(
                [self.label_to_id_map[label]]
                + [self.pad_token_label_id] * (len(subword_tokens) - 1)
            )

        text_field = TextField(tokens, self.token_indexers)
        fields = {"text": text_field}

        fields["labels"] = SequenceLabelField(label_ids, text_field)
        instance = Instance(fields)

        metadata = dict(guid=document.guid)

        feature = InputFeaturesAllennlp(
            instance=instance,
            metadata=metadata,
        )

        # TODO: implement!
        # if verbose:
        #     self._log_input_features(tokens, document, features, labels)

        return [feature]


    def documents_to_features(
        self, documents: List[Document]
    ) -> List[InputFeatures]:
        input_features = []
        num_shown_input_features = 0
        for doc_idx, document in enumerate(documents):
            if doc_idx % 10000 == 0:
                logger.info("Writing document %d of %d" % (doc_idx, len(documents)))

            verbose = num_shown_input_features < self.log_num_input_features
            doc_input_features = self.document_to_features(document, verbose)
            input_features.extend(doc_input_features)
            num_shown_input_features += 1

        # logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
        num_fit_examples = len(documents) - sum(
            [features.metadata["truncated"] for features in input_features]
        )
        logger.info(
            "%d (%.2f %%) examples can fit max_seq_length = %d"
            % (num_fit_examples, num_fit_examples * 100.0 / len(documents), self.max_length)
        )

        return input_features
