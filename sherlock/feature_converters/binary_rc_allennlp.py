import itertools
import logging
from typing import List, Optional, Tuple

from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import Tokenizer, PreTrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from sherlock.feature_converters.feature_converter import InputFeatures, FeatureConverter
from sherlock.feature_converters.feature_converter_allennlp import FeatureConverterAllennlp

from sherlock import Document
from sherlock.feature_converters.input_features import InputFeaturesAllennlp

logger = logging.getLogger(__name__)


@FeatureConverter.register("binary_rc_allennlp")
class BinaryRcConverterAllennlp(FeatureConverterAllennlp):
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexer: TokenIndexer,
        vocab: Vocabulary,
        labels: List[str],
        max_length: int=512,
        entity_handling: str="mark_entity",
        log_num_input_features: int=-1,
        sep_token: str=None,
    ) -> None:
        super().__init__(tokenizer, token_indexer, vocab, labels, max_length)
        if entity_handling not in [
            "mark_entity",
            "mark_entity_append_ner",
            "mask_entity",
            "mask_entity_append_text",
        ]:
            raise ValueError("Unknown entity handling '%s'." % entity_handling)

        self.entity_handling = entity_handling
        self.log_num_input_features = log_num_input_features

        if isinstance(tokenizer, PreTrainedTransformerTokenizer):
            if sep_token is None:
                # access transformers sep_token automatically:
                # https://github.com/huggingface/transformers/blob/b66c5ab20c8bb08d52cb840382498f936ea8da03/src/transformers/tokenization_utils_base.py#L985
                self.sep_token = tokenizer.tokenizer.sep_token
            else:
                # Overriding native sep_token <- undefined behavior
                if sep_token != tokenizer.tokenizer.sep_token:
                    logger.warn("Overwriting transformer sep_token leads to undefined behavior!")
                self.sep_token = sep_token
        else:
            if sep_token is None:
                # Option 1: people need to give the sep_token: TODO: decide
                return NotImplementedError(
                    "FeatureConverterAllennlp for non-transformers must specify sep_token")
                # Option 2: set sep_token for people
                self.sep_token = "[SEP]"
            else:
                self.sep_token = sep_token

    @property
    def sep_token(self):
        if self.sep_token:
            return self.sep_token
        # Option 1: let people handle their sep_token themselves

    @property
    def name(self) -> str:
        return "binary_rc_allennlp"

    @property
    def persist_attributes(self) -> List[str]:
        return ["max_length", "entity_handling"]

    def document_to_features(
        self, document: Document, verbose: bool = False
    ) -> List[InputFeatures]:
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

        input_features = []
        for head_idx, tail_idx, label, sent_id in mention_combinations:
            tokens = self._handle_entities(document, head_idx, tail_idx, sent_id)

            text_field = TextField(tokens, self.token_indexer)
            fields = {"text": text_field}
            if label is not None:
                label_id = self.label_to_id_map[label]
                label_field = LabelField(label)
                fields["label"] = label_field
            instance = Instance(fields)
            instance.index_fields(self.vocab)

            metadata = dict(
                guid=document.guid,
                head_idx=head_idx,
                tail_idx=tail_idx,
            )

            features = InputFeaturesAllennlp(
                instance=instance,
                metadata=metadata,
            )
            input_features.append(features)

            # if verbose: # TODO: implement for allennlp InputFeaturesAllennlp
            #     self._log_input_features(tokens, document, features, label)

        return input_features

    def documents_to_features(self, documents: List[Document]) -> List[InputFeatures]:
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

    # Tokenize given Document/Sentence, handle head and tail entity
    # according to entity_handling strategy
    def _handle_entities(
        self, document: Document, head_idx: int, tail_idx: int, sent_idx: Optional[int] = None
    ) -> List[Token]:
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
            # self.entity_handling.startswith("mask_entity") case
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
        return self.tokenizer.tokenize(" ".join(tokens))
