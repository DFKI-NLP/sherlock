import itertools
import logging
from typing import List, Optional, Tuple

from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer
from transformers import PreTrainedTokenizer

from sherlock import Document
from sherlock.feature_converters.feature_converter import FeatureConverter
from sherlock.feature_converters.input_features import (
    InputFeatures, InputFeaturesAllennlp, InputFeaturesTransformers)


logger = logging.getLogger(__name__)


@FeatureConverter.register("binary_rc")
class BinaryRcConverter(FeatureConverter):
    """
    Class to convert Documents into InputFeatures used for Annotators
    in Binary Relation Classification.

    Attributes
    ----------
    labels
    max_length
    framework
    entity_handling
    pad_token_segment_id    # TODO: probably move into `transformers` kwargs
    log_num_input_features
    kwargs : ``Dict[str, any]``
        framwork specific keywords.
        `transformers`:
            tokenizer  : `PreTrainedTokenizer`
        `allennlp`:
            tokenizer : ``Tokenizer``
            token_indexers : ``Dict[str,TokenIndexer]``
            sep_token : ``str``, optional (default=`None`)
                model-specific separator token used to separate
                relations at the end of sentence. Is automatically
                set for allennlp-transformers, but for other models
                has to be set to separator used in training.
    """
    def __init__(
        self,
        labels: List[str],
        max_length: int = 512,
        framework: str = "transformers",
        entity_handling: str = "mark_entity",
        pad_token_segment_id: int = 0,      # TODO: Remove? Never used.
        log_num_input_features: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(labels, max_length, framework, **kwargs)
        if entity_handling not in [
            "mark_entity",
            "mark_entity_append_ner",
            "mask_entity",
            "mask_entity_append_text",
        ]:
            raise ValueError("Unknown entity handling '%s'." % entity_handling)

        self.entity_handling = entity_handling
        self.pad_token_segment_id = pad_token_segment_id
        self.log_num_input_features = log_num_input_features

        if framework == "transformers":
            sep_token = kwargs.get("sep_token")
            if sep_token is None:
                self.sep_token = self.tokenizer.sep_token
            else:
                # Overriding native sep_token <- undefined behavior
                if sep_token != self.tokenizer.sep_token:
                    logger.warn("Overwriting transformer sep_token leads to undefined behavior!")
                self.sep_token = sep_token

            # Some tokenizer lowercase or not, to compare the head/tail tokens
            # later this seems to be the easiest way to do it
            lower_cases = "a" in " ".join(self.tokenizer.tokenize("A"))

        elif framework == "allennlp":
            sep_token = kwargs.get("sep_token")
            if isinstance(self.tokenizer, PretrainedTransformerTokenizer):
                if sep_token is None:
                    # access transformers sep_token automatically:
                    # https://github.com/huggingface/transformers/blob/b66c5ab20c8bb08d52cb840382498f936ea8da03/src/transformers/tokenization_utils_base.py#L985
                    self.sep_token = self.tokenizer.tokenizer.sep_token
                else:
                    # Overriding native sep_token <- undefined behavior
                    if sep_token != self.tokenizer.tokenizer.sep_token:
                        logger.warn("Overwriting transformer sep_token leads to undefined behavior!")
                    self.sep_token = sep_token
            else:
                if sep_token is None:
                    self.sep_token = "[SEP]"
                    logger.warn(
                        f"sep_token not given, setting it to {self.sep_token}."
                        + "If the model was trained without this token this can"
                        + "lead to undefined behavior.")
                else:
                    self.sep_token = sep_token

            lower_cases = "a" in " ".join(
                [token.text for token in self.tokenizer.tokenize("A")])

        if lower_cases:
            self.marker_tokens = [
                "[head_start]","[head_end]", "[tail_start]", "[tail_end]"]
        else:
            self.marker_tokens = [
                "[HEAD_START]","[HEAD_END]", "[TAIL_START]", "[TAIL_END]"]

    @property
    def name(self) -> str:
        return "binary_rc"

    @property
    def persist_attributes(self) -> List[str]:
        if self.framework == "transformers":
            return ["max_length", "entity_handling", "pad_token_segment_id", "sep_token"]
        elif self.framework == "allennlp":
            return ["max_length", "entity_handling", "pad_token_segment_id", "sep_token"]

    def document_to_features_transformers(
        self, document: Document, verbose: bool = False
    ) -> List[InputFeaturesTransformers]:

        assert isinstance(self.tokenizer, PreTrainedTokenizer),\
            "FeatureConverter initialized with wrong Tokenizer class"

        mention_combinations = self._create_mention_combinations(document)

        input_features = []
        for head_idx, tail_idx, label, sent_id in mention_combinations:
            input_string = self._handle_entities(document, head_idx, tail_idx, sent_id)
            tokens = self.tokenizer.tokenize(input_string)

            print(input_string)
            print(tokens)

            inputs = self.tokenizer.encode_plus(
                text=tokens,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_overflowing_tokens=True,
            )

            metadata = dict(
                guid=document.guid,
                truncated="overflowing_tokens" in inputs and len(inputs["overflowing_tokens"]) > 0,
                head_idx=head_idx,
                tail_idx=tail_idx,
            )

            label_id = self.label_to_id_map[label] if label is not None else None

            features = InputFeaturesTransformers(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                labels=label_id,
                metadata=metadata,
            )
            input_features.append(features)

            if verbose:
                self._log_input_features(tokens, document, features, label)

        return input_features

    def document_to_features_allennlp(
        self, document: Document, verbose: bool=False
    ) -> List[InputFeaturesAllennlp]:

        assert isinstance(self.tokenizer, Tokenizer),\
            "FeatureConverter initialized with wrong Tokenizer class"

        mention_combinations = self._create_mention_combinations(document)

        if verbose:
            logger.warning("Logging function for Allennlp FeatureConverter not implemented yet")

        input_features = []
        for head_idx, tail_idx, label, sent_id in mention_combinations:
            input_string = self._handle_entities(document, head_idx, tail_idx, sent_id)

            tokens = self.tokenizer.tokenize(input_string)

            print(tokens)

            # If head or tail have been cutoff: ignore this Instance
            # There is no good way to know whether the tokenizer has cutoff
            # the instance itself or if the indices of head/tail stayed the
            # same in the tokenized sequence: thus have to loop through tokens
            # TODO: talk to Leo about this

            # marker_counter = 0

            # TODO: different modes of entity masking.
            # print(tokens)

            # if self.framework == "allennlp":
            #     for token in tokens:
            #         if token.text in self.marker_tokens:
            #             marker_counter += 1
            # elif self.framework == "transformers":
            #     for token in tokens:
            #         if token in self.marker_tokens:
            #             marker_counter += 1
            # else:
            #     raise NotImplementedError(
            #         "Framwork needs to implement marker_token counter to determine cutoff")

            # if marker_counter < 4:
            #     # Head/Tail token was cut off
            #     continue
            # elif marker_counter > 4:
            #     logger.critical(
            #         f"Input string has more than 4 entity marker tokens:\n"
            #         + "Check whether Tokenizer is tokenizing properly."
            #         + f"Tokens: {tokens}\n"
            #         + f"Input String: {input_string}"
            #     )

            text_tokens_field = TextField(tokens[:self.max_length],
                                          self.token_indexers)

            fields = {"text": text_tokens_field}

            if label is not None:
                label_field = LabelField(label)
                # skip_indexing=True leads to allennlp not creating a vocabulary
                # for labels. Then vocabulary.get_vocab_size("labels") will be 0
                # but some models use vocabulary.get_vocab_size("labels")
                # For that reason labels are given as strings to the vocabulary
                # so that allennlp can handle it in its own way.
                fields["label"] = label_field
            #if instance_id is not None:
            #    fields["metadata"]["id"] = instance_id
            instance = Instance(fields)

            # Cannot know for sure if tokenizer truncated, use heuristic:
            # if tokens is max_length, it is very likely that truncation
            # happened. TODO: dicuss with Leo

            metadata = dict(
                guid=document.guid,
                truncated=len(tokens) >= self.max_length,
                head_idx=head_idx,
                tail_idx=tail_idx,
            )

            features = InputFeaturesAllennlp(
                instance=instance,
                metadata=metadata,
            )
            input_features.append(features)

        return input_features


    def _create_mention_combinations(self, document: Document) -> List[str]:
        """
        Converts Document to List of InputFeatures usable to train
        """

        # List of Tuples head_idx, tail_idx, relation_label, sentence_idx
        mention_combinations: List[Tuple[int, int, Optional[str], Optional[int]]] = []

        # If relations are present, use them
        if document.rels:
            for relation in document.rels:
                mention_combinations.append(
                    (relation.head_idx, relation.tail_idx, relation.label, None)
                )
        else:
            # No relations -> create combinations between Mentions within
            # sentences
            if document.sents:
                for sent_idx, sent in enumerate(document.sents):
                    sent_ments = [
                        idx
                        for idx, ment in enumerate(document.ments)
                        if sent.start <= ment.start < sent.end
                    ]
                    for head_idx, tail_idx in itertools.product(sent_ments, repeat=2):
                        if head_idx == tail_idx:
                            continue
                        mention_combinations.append((head_idx, tail_idx, None, sent_idx))
            else:
                # No sentences -> create combinations between all Mentions
                for head_idx, tail_idx in itertools.product(range(len(document.ments)), repeat=2):
                    if head_idx == tail_idx:
                        continue
                    mention_combinations.append((head_idx, tail_idx, None, None))

        return mention_combinations


    def documents_to_features(self, documents: List[Document]) -> List[InputFeatures]:
        input_features = []
        num_shown_input_features = 0
        for doc_idx, document in enumerate(documents):
            if doc_idx % 10000 == 0:
                logger.info("Converting document %d of %d to features" % (doc_idx, len(documents)))

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


    def _handle_entities(
        self, document: Document, head_idx: int, tail_idx: int, sent_idx: Optional[int] = None
    ) -> str:
        """Apply entity handling strategy on Document and return string
        ready to be tokenized.
        """

        head_mention = document.ments[head_idx]
        tail_mention = document.ments[tail_idx]

        ner_head = "[HEAD=%s]" % head_mention.label
        ner_tail = "[TAIL=%s]" % tail_mention.label

        sep_token = self.sep_token

        # Limit search space for known sentence id
        if sent_idx is None:
            input_tokens = document.tokens
        else:
            sent = document.sents[sent_idx]
            input_tokens = document.tokens[sent.start : sent.end]

        tokens = []
        if self.entity_handling.startswith("mark_entity"):
            for i, token in enumerate(input_tokens):
                if i == head_mention.start:
                    tokens.append("[HEAD_START]")
                if i == tail_mention.start:
                    tokens.append("[TAIL_START]")
                if i == head_mention.end:
                    tokens.append("[HEAD_END]")
                if i == tail_mention.end:
                    tokens.append("[TAIL_END]")
                tokens.append(token.text)
            if self.entity_handling == "mark_entity_append_ner":
                tokens = tokens + [sep_token, ner_head, sep_token, ner_tail]
        else:
            # "mask_entity" case
            # Collect head_tokens, tail_tokens and other tokens separately
            head_tokens = []
            tail_tokens = []
            for i, token in enumerate(input_tokens):
                if i == head_mention.start:
                    tokens.append(ner_head)
                if i == tail_mention.start:
                    tokens.append(ner_tail)
                if (i >= head_mention.start) and (i < head_mention.end):
                    head_tokens.append(token.text)
                elif (i >= tail_mention.start) and (i < tail_mention.end):
                    tail_tokens.append(token.text)
                else:
                    tokens.append(token.text)
            if self.entity_handling == "mask_entity_append_text":
                tokens.append(sep_token)
                tokens.extend(head_tokens)
                tokens.append(sep_token)
                tokens.extend(tail_tokens)
        return " ".join(tokens)