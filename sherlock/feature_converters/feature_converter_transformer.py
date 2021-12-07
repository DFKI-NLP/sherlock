import os
import json
import logging
from typing import List, Optional, Union

from transformers import PreTrainedTokenizer

from sherlock.document import Document
from sherlock.feature_converters.input_features import InputFeatures
from sherlock.feature_converters.feature_converter import FeatureConverter


logger = logging.getLogger(__name__)


class FeatureConverterTransformer(FeatureConverter):
    """
    Abstract class for FeatureConverters to convert Documents into
    representation usable for (huggingface) transformers.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        max_length: int = 512,
    ) -> None:
        super().__init__(labels, max_length)
        self.tokenizer = tokenizer

    @staticmethod
    def from_pretrained(path: str, tokenizer: PreTrainedTokenizer) -> "FeatureConverter":
        vocab_file = os.path.join(path, "converter_label_vocab.txt")
        converter_config_file = os.path.join(path, "converter_config.json")
        with open(converter_config_file, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        with open(vocab_file, "r", encoding="utf-8") as reader:
            config["labels"] = [line.strip() for line in reader.readlines()]
        config["tokenizer"] = tokenizer
        converter_class = FeatureConverter.by_name(config.pop("name"))
        return converter_class(**config)

    def _log_input_features(
        self,
        tokens: List[str],
        document: Document,
        features: InputFeatures,
        labels: Optional[Union[str, List[str]]] = None,
    ) -> None:
        logger.info("*** Example ***")
        logger.info("guid: %s", document.guid)
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids: %s", " ".join([str(x) for x in features.input_ids]))
        logger.info("attention_mask: %s", " ".join([str(x) for x in features.attention_mask]))
        if features.token_type_ids is not None:
            logger.info("token_type_ids: %s", " ".join([str(x) for x in features.token_type_ids]))
        if labels:
            logger.info("labels: %s (ids = %s)", labels, features.labels)
