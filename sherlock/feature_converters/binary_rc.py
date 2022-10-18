# -*- coding: utf8 -*-
"""

@date: 09.02.22
@author: christoph.alt@posteo.de, gabriel.kressin@dfki.de, leonhard.hennig@dfki.de
"""
import itertools
import logging
from typing import List, Optional, Tuple, Union

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import (
    Tokenizer, PretrainedTransformerTokenizer, Token)
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

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
    max_length : ``int``, optional (default=`512`)
        Will limit sequences of tokens to maximum length.
    framework : ``"allennlp" | "transformers"``, optional (default=`transformers`)
        String indicating to use allennlp or transformers library.
    entity_handling : ``str``, optional (default=`mark_entity`)
        Strategy to specifically mark entities in sentences. Has to be between
        `"mark_entity", "mark_entity_append_ner", "mask_entity", "mask_entity_append_text"`.
        unclear what this does.
    log_num_input_features : ``int``, optional (default=`-1`)
        Amount of example Instances which are logged.
    tokenize_special_tokens : ``bool``, optional (default=`None`)
        Flag to decide whether to tokenize special tokens or not: if set to
        `None` will determine automatically what to do.
        This flag is needed because some tokenizers in allennlp split the
        [head_start]/[head_end] tokens (within eachother, e.g. SpacyTokenizer)
        without the option to make exceptions for some tokens. This requires
        not tokenizing the special tokens at.
        On the other hand some other tokenizers (e.g. Transformer tokenizer)
        *require* the tokenizer to run on the tokens, because they (secretely)
        do the indexing as well.
    kwargs : ``Dict[str, any]``
        Framework specific keywords.
        `transformers`:
            labels : ``List[str]``
                List of all labels as strings.
            tokenizer  : ``PreTrainedTokenizer``
                Huggingface tokenizer to tokenize input-sentences.
        `allennlp`:
            tokenizer : ``Tokenizer``
                AllenNLP tokenizer to tokenize input-sentences.
            token_indexers : ``Dict[str,TokenIndexer]``
                AllenNLP token indexer to index vocabulary with.
            sep_token : ``str``, optional (default=`None`)
                model-specific separator token used to separate
                relations at the end of sentence. Is automatically
                set for allennlp-transformers, but for other models
                has to be set to separator used in training.
    """

    def __init__(
            self,
            max_length: Optional[int] = None,
            framework: str = "transformers",
            entity_handling: str = "mark_entity",
            log_num_input_features: int = -1,
            tokenize_special_tokens: Optional[bool] = None,
            **kwargs,
    ) -> None:
        super().__init__(max_length, framework, **kwargs)
        if entity_handling not in [
            "mark_entity",
            "mark_entity_append_ner",
            "mask_entity",
            "mask_entity_append_text",
        ]:
            raise ValueError("Unknown entity handling '%s'." % entity_handling)

        self.entity_handling = entity_handling
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

            if tokenize_special_tokens is not None:
                self.tokenize_special_tokens = tokenize_special_tokens
            else:
                self.tokenize_special_tokens = True

            tokenizer_test_string = " ".join(self.tokenizer.tokenize("A"))
            self.n_special_tokens = self.tokenizer.num_special_tokens_to_add()

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

                if tokenize_special_tokens is not None:
                    if not tokenize_special_tokens:
                        logger.warn(
                            "Not tokenizing special tokens with PretrainedTransformerTokenizer"
                            + "can lead to problems later. Consider unsetting"
                            + "`tokenize_special_tokens` in `feature_converter.BinaryRcConverter`"
                            + " arguments."
                        )
                    self.tokenize_special_tokens = tokenize_special_tokens
                else:
                    self.tokenize_special_tokens = True
            else:
                if sep_token is None:
                    self.sep_token = "[SEP]"
                    logger.warn(
                        f"sep_token not given, setting it to {self.sep_token}."
                        + "If the model was trained without this token this can"
                        + "lead to undefined behavior.")
                else:
                    self.sep_token = sep_token

                if tokenize_special_tokens is None:
                    self.tokenize_special_tokens = False
                else:
                    self.tokenize_special_tokens = tokenize_special_tokens

            tokenizer_test_tokens = [token.text for token in self.tokenizer.tokenize("A")]
            tokenizer_test_string = " ".join(
                tokenizer_test_tokens
            )
            # Need to make sure tokenizer does not add special tokens.
            special_tokens = self.tokenizer.add_special_tokens([])
            for special_token in special_tokens:
                if special_token.text in tokenizer_test_string:
                    raise ConfigurationError(
                        f"Tokenizer adds special token `{special_token}`, but "
                        + "is not allowed to do so. Make sure `add_special_tokens`"
                        + "is set to `False` in Tokenizer initialization."
                    )

            self.n_special_tokens = len(special_tokens)

        # Need to find out whether the tokenizer lowercases.
        self.lower_cases = "a" in tokenizer_test_string

        if self.lower_cases:
            self.marker_tokens = [
                "[head_start]", "[head_end]", "[tail_start]", "[tail_end]"]
        else:
            self.marker_tokens = [
                "[HEAD_START]", "[HEAD_END]", "[TAIL_START]", "[TAIL_END]"]

    @property
    def name(self) -> str:
        return "binary_rc"

    @property
    def persist_attributes(self) -> List[str]:
        if self.framework == "transformers":
            return ["max_length", "entity_handling", "sep_token"]
        elif self.framework == "allennlp":
            return ["max_length", "entity_handling", "sep_token"]

    def document_to_features_transformers(
            self, document: Document, verbose: bool = False
    ) -> List[InputFeaturesTransformers]:

        assert isinstance(self.tokenizer, PreTrainedTokenizer) or isinstance(self.tokenizer, PreTrainedTokenizerFast), \
            f"FeatureConverter initialized with wrong Tokenizer class: {self.tokenizer.__class__}"

        mention_combinations = self._create_mention_combinations(document)

        input_features = []
        for head_idx, tail_idx, label, sent_id in mention_combinations:
            tokens, entity_cutoff, truncated = self._tokenize_with_entities(
                document, head_idx, tail_idx, sent_id
            )
            # If head or tail have been cutoff: ignore this Instance
            if entity_cutoff:
                continue

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
                truncated=truncated,
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
            self, document: Document, verbose: bool = False
    ) -> List[InputFeaturesAllennlp]:

        assert isinstance(self.tokenizer, Tokenizer), \
            "FeatureConverter initialized with wrong Tokenizer class"

        mention_combinations = self._create_mention_combinations(document)

        input_features = []
        for head_idx, tail_idx, label, sent_id in mention_combinations:
            tokens, entity_cutoff, truncated = self._tokenize_with_entities(
                document, head_idx, tail_idx, sent_id
            )
            # If head or tail have been cutoff: ignore this Instance
            if entity_cutoff:
                continue

            # Add special tokens
            tokens = self.tokenizer.add_special_tokens(tokens)

            text_tokens_field = TextField(tokens,
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

            instance = Instance(fields)

            metadata = dict(
                guid=document.guid,
                truncated=truncated,
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
                    for head_idx, tail_idx in itertools.permutations(sent_ments, r=2):
                        if head_idx == tail_idx:
                            continue
                        mention_combinations.append((head_idx, tail_idx, None, sent_idx))
            else:
                # No sentences -> create combinations between all Mentions
                for head_idx, tail_idx in itertools.permutations(range(len(document.ments)), r=2):
                    if head_idx == tail_idx:
                        continue
                    mention_combinations.append((head_idx, tail_idx, None, None))

        return mention_combinations

    def documents_to_features(self, documents: List[Document]) -> List[InputFeatures]:
        input_features: List[InputFeatures] = []
        num_shown_input_features = 0
        for doc_idx, document in enumerate(documents):
            if doc_idx % 10000 == 0:
                logger.info("Converting document %d of %d to features" % (doc_idx, len(documents)))

            verbose = num_shown_input_features < self.log_num_input_features
            doc_input_features = self.document_to_features(document, verbose)
            input_features.extend(doc_input_features)
            num_shown_input_features += 1
        logger.info("Converted all documents.")

        # logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
        n_fit_examples = len(input_features)
        percentage = 100 * n_fit_examples / len(documents)
        n_truncated_without_entity_cutoff = sum((
            1 for features in input_features if features.metadata["truncated"]
        ))
        n_untruncated = n_fit_examples - n_truncated_without_entity_cutoff
        percentage_untruncated = 100 * n_untruncated / len(documents)
        logger.info(
            f"{n_fit_examples} ({percentage:.2f} %) examples can fit max_seq_length = {self.max_length}"
            + " without entity cutoff."
        )
        logger.info(
            f"{n_untruncated} ({percentage_untruncated:.2f} % ) examples can fit"
            + f" max_seq_length = {self.max_length} without truncation."
        )
        return input_features

    def _handle_special_token(self, token: str) -> Union[List[str], List[Token]]:
        if self.tokenize_special_tokens:
            return self.tokenizer.tokenize(token)
        if self.framework == "allennlp":
            # Need to return Token class for allennlp
            return [Token(text=token), ]
        return [token, ]

    def _handle_special_tokens(self, tokens: List[str]) -> Union[List[str], List[Token]]:
        return list(
            itertools.chain.from_iterable(
                [self._handle_special_token(token) for token in tokens]
            )
        )

    def _check_truncated_entity(self, tokens: Union[List[str], List[Token]]):
        if self.max_length:
            return len(tokens) + self.n_special_tokens > self.max_length
        return False

    def _tokenize_with_entities(
            self,
            document: Document,
            head_idx: int,
            tail_idx: int,
            sent_idx: Optional[int] = None,
    ) -> Tuple[Union[List[str], List[Token]], bool, bool]:
        """Apply entity handling strategy on Document and tokenize
        text.

        Returns
        -------
        tokens : str
            tokenized tokens.
        truncated_entity : bool
            whether an entity was truncated.
        truncated : bool
            whether tokens have been truncated in general
        """
        # assert no special tokens are added
        # check transformers implementation

        head_mention = document.ments[head_idx]
        tail_mention = document.ments[tail_idx]

        ner_head = f"[HEAD={head_mention.label}]"
        ner_tail = f"[TAIL={tail_mention.label}]"
        if self.lower_cases:
            ner_head = ner_head.lower()
            ner_tail = ner_tail.lower()

        sep_token = self.sep_token

        # Limit search space for known sentence id
        if sent_idx is None:
            input_tokens = document.tokens
        else:
            sent = document.sents[sent_idx]
            input_tokens = document.tokens[sent.start: sent.end]

        truncated_entity = False
        tokens = []
        # Save untokenized tokens in temporary list
        temporary = []
        if self.entity_handling.startswith("mark_entity"):
            for i, token in enumerate(input_tokens):
                if i == head_mention.start:
                    # tokenize temporary list jointly
                    tokens.extend(self.tokenizer.tokenize(" ".join(temporary)))
                    # clear temporary for next segment
                    temporary = []
                    # add segment to tokenized list
                    tokens.extend(self._handle_special_token(self.marker_tokens[0]))
                    # check if tokens will be truncated
                    truncated_entity = self._check_truncated_entity(tokens)
                if i == tail_mention.start:
                    tokens.extend(self.tokenizer.tokenize(" ".join(temporary)))
                    temporary = []
                    tokens.extend(self._handle_special_token(self.marker_tokens[2]))
                    truncated_entity = self._check_truncated_entity(tokens)
                if i == head_mention.end:
                    tokens.extend(self.tokenizer.tokenize(" ".join(temporary)))
                    temporary = []
                    tokens.extend(self._handle_special_token(self.marker_tokens[1]))
                    truncated_entity = self._check_truncated_entity(tokens)
                if i == tail_mention.end:
                    tokens.extend(self.tokenizer.tokenize(" ".join(temporary)))
                    temporary = []
                    tokens.extend(self._handle_special_token(self.marker_tokens[3]))
                    truncated_entity = self._check_truncated_entity(tokens)
                temporary.append(token.text)
            if len(temporary):
                tokens.extend(self.tokenizer.tokenize(" ".join(temporary)))
            if self.entity_handling == "mark_entity_append_ner":
                tokens.extend(
                    self._handle_special_tokens(
                        [sep_token, ner_head, sep_token, ner_tail]
                    )
                )
                truncated_entity = self._check_truncated_entity(tokens)
        else:
            # "mask_entity" case
            # Collect head_tokens, tail_tokens and other tokens separately
            head_tokens = []
            tail_tokens = []
            for i, token in enumerate(input_tokens):
                if i == head_mention.start:
                    tokens.extend(self.tokenizer.tokenize(" ".join(temporary)))
                    temporary = []
                    tokens.extend(self._handle_special_token(ner_head))
                    truncated_entity = self._check_truncated_entity(tokens)
                if i == tail_mention.start:
                    tokens.extend(self.tokenizer.tokenize(" ".join(temporary)))
                    temporary = []
                    tokens.extend(self._handle_special_token(ner_tail))
                    truncated_entity = self._check_truncated_entity(tokens)
                if (i >= head_mention.start) and (i < head_mention.end):
                    head_tokens.append(token.text)
                elif (i >= tail_mention.start) and (i < tail_mention.end):
                    tail_tokens.append(token.text)
                else:
                    temporary.append(token.text)
            if len(temporary):
                tokens.extend(self.tokenizer.tokenize(" ".join(temporary)))
            if self.entity_handling == "mask_entity_append_text":
                tokens.extend(self._handle_special_token(sep_token))
                tokens.extend(self.tokenizer.tokenize(" ".join(head_tokens)))
                tokens.extend(self._handle_special_token(sep_token))
                tokens.extend(self.tokenizer.tokenize(" ".join(tail_tokens)))
                truncated_entity = self._check_truncated_entity(tokens)

        if self.max_length:
            truncated = len(tokens) > self.max_length
            return tokens[:self.max_length - self.n_special_tokens], truncated_entity, truncated
        else:
            return tokens, truncated_entity, False
