import json
import logging
import os
from typing import List

from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer, XLNetTokenizer

from sherlock import Document
from sherlock.feature_converters.feature_converter import (
    FeatureConverter,
    InputFeatures,
)


logger = logging.getLogger(__name__)


class NerConverter(FeatureConverter):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        max_length: int = 512,
        pad_token_segment_id: int = 0,
        pad_token_label_id: int = CrossEntropyLoss().ignore_index,
        log_num_input_features: int = -1,
    ) -> None:
        super().__init__(tokenizer, labels, max_length)
        self.pad_token_segment_id = pad_token_segment_id
        self.pad_token_label_id = pad_token_label_id
        self.log_num_input_features = log_num_input_features

    def save(self, save_directory: str) -> None:
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
        self.save_vocabulary(save_directory)
        config = dict(
            max_length=self.max_length,
            pad_token_segment_id=self.pad_token_segment_id,
            pad_token_label_id=self.pad_token_label_id,
        )
        converter_config_file = os.path.join(save_directory, "converter_config.json")
        with open(converter_config_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(config, ensure_ascii=False))

    def document_to_features(
        self, document: Document, verbose: bool = False
    ) -> List[InputFeatures]:
        tokens = []  # type: List[str]
        labels = []  # type: List[str]
        label_ids = []  # type: List[int]
        for token in document.tokens:
            subword_tokens = self.tokenizer.tokenize(token.text)
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
            max_length=self.max_length,
            pad_to_max_length=True,
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

        features = InputFeatures(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            labels=label_ids,
            metadata=metadata,
        )

        if verbose:
            self._log_input_features(tokens, document, features, labels)

        return [features]

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