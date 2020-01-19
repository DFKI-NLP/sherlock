from typing import List, Tuple, Optional

import os
import json
import logging
import itertools

from transformers import PreTrainedTokenizer
from sherlock import Document
from sherlock.feature_converters import FeatureConverter, InputFeatures

logger = logging.getLogger(__name__)


class BinaryRelationClfConverter(FeatureConverter):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 labels: List[str],
                 max_length: int = 512,
                 entity_handling: str = "mark_entity",
                 pad_on_left: bool = False,
                 pad_token_id: int = 0,
                 pad_token_segment_id: int = 0,
                 mask_padding_with_zero: bool = True,
                 log_num_input_features: int = -1) -> None:
        super().__init__(tokenizer, labels, max_length)
        if entity_handling not in ["mark_entity", "mark_entity_append_ner",
                                   "mask_entity", "mask_entity_append_text"]:
            raise ValueError("Unknown entity handling '%s'." % entity_handling)

        self.entity_handling = entity_handling
        self.pad_on_left = pad_on_left
        self.pad_token_id = pad_token_id
        self.pad_token_segment_id = pad_token_segment_id
        self.mask_padding_with_zero = mask_padding_with_zero
        self.log_num_input_features = log_num_input_features

    @classmethod
    def from_pretrained(cls,
                        path: str,
                        tokenizer: PreTrainedTokenizer) -> "BinaryRelationClfConverter":
        vocab_file = os.path.join(path, "converter_label_vocab.txt")
        converter_config_file = os.path.join(path, "converter_config.json")
        with open(converter_config_file, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        with open(vocab_file, "r", encoding="utf-8") as reader:
            config["labels"] = [line.strip() for line in reader.readlines()]
        config["tokenizer"] = tokenizer
        return cls(**config)

    def save(self, save_directory: str) -> None:
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
        self.save_vocabulary(save_directory)
        config = dict(max_length=self.max_length,
                      entity_handling=self.entity_handling,
                      pad_on_left=self.pad_on_left,
                      pad_token_id=self.pad_token_id,
                      pad_token_segment_id=self.pad_token_segment_id,
                      mask_padding_with_zero=self.mask_padding_with_zero)
        converter_config_file = os.path.join(save_directory, "converter_config.json")
        with open(converter_config_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(config, ensure_ascii=False))

    def save_vocabulary(self, vocab_path: str) -> None:
        """Save the converters label vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, "converter_label_vocab.txt")
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for label, label_index in self.label_to_id_map.items():
                if index != label_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = label_index
                writer.write(label + "\n")
                index += 1

    def document_to_features(self,
                             document: Document,
                             verbose: bool = False) -> List[InputFeatures]:
        """
        Loads a data file into a list of ``InputFeatures``
        Args:
            examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
            tokenizer: Instance of a tokenizer that will tokenize the examples
            label_list: List of labels
            max_length: Maximum example length, 512 by default
            entity_handling: mark_entity, mark_entity_append_ner, mask_entity, mask_entity_append_text
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token_id: Padding token
            pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)
        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.
        """

        mention_combinations = []  # type: List[Tuple[int, int, Optional[str]]]
        if document.rels:
            for relation in document.rels:
                mention_combinations.append((relation.head_idx, relation.tail_idx, relation.label))
        else:
            for head_idx, tail_idx in itertools.product(range(len(document.ments)), repeat=2):
                if head_idx == tail_idx:
                    break
                mention_combinations.append((head_idx, tail_idx, None))

        input_features = []
        for head_idx, tail_idx, label in mention_combinations:
            tokens = self._handle_entities(document, head_idx, tail_idx)

            inputs = self.tokenizer.encode_plus(
                text=tokens,
                add_special_tokens=True,
                max_length=self.max_length,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            metadata = dict(truncated="overflowing_tokens" in inputs,
                            head_idx=head_idx,
                            tail_idx=tail_idx)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.max_length - len(input_ids)
            if self.pad_on_left:
                input_ids = ([self.pad_token_id] * padding_length) + input_ids
                attention_mask = ([0 if self.mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([self.pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([self.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([self.pad_token_segment_id] * padding_length)

            assert len(input_ids) == self.max_length, "Error with input length {} vs {}".format(len(input_ids), self.max_length)
            assert len(attention_mask) == self.max_length, "Error with input length {} vs {}".format(len(attention_mask), self.max_length)
            assert len(token_type_ids) == self.max_length, "Error with input length {} vs {}".format(len(token_type_ids), self.max_length)

            if label is not None:
                label_id = self.label_to_id_map[label]

            features = InputFeatures(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     labels=label_id,
                                     metadata=metadata)
            input_features.append(features)

            if verbose:
                self._log_input_features(tokens, document, features, label)

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

        # logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
        num_fit_examples = len(documents) - sum([features.metadata["truncated"]
                                                 for features in input_features])
        logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                    num_fit_examples * 100.0 / len(documents), self.max_length))

        return input_features

    def _log_input_features(self,
                            tokens: List[str],
                            document: Document,
                            features: InputFeatures,
                            label: str = None) -> None:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (document.guid))
        logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in features.input_ids]))
        logger.info("attention_mask: %s" % " ".join([str(x) for x in features.attention_mask]))
        logger.info("token_type_ids: %s" % " ".join([str(x) for x in features.token_type_ids]))
        if label:
            logger.info("label: %s (id = %d)" % (label, features.labels))

    def _handle_entities(self,
                         document: Document,
                         head_idx: int,
                         tail_idx: int) -> List[str]:
        head_mention = document.ments[head_idx]
        tail_mention = document.ments[tail_idx]

        ner_head = "[HEAD=%s]" % head_mention.label
        ner_tail = "[TAIL=%s]" % tail_mention.label

        sep_token = self.tokenizer.sep_token

        tokens = []
        if self.entity_handling.startswith("mark_entity"):
            for i, token in enumerate(document.tokens):
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
            head_tokens = []
            tail_tokens = []
            for i, token in enumerate(document.tokens):
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