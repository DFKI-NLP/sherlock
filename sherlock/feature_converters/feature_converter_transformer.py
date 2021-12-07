import json
import os
from typing import List

from transformers import PreTrainedTokenizer

from sherlock.feature_converters.feature_converter import FeatureConverter


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
