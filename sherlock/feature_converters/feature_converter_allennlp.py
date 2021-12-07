import json
import os
from typing import List

from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, token_indexer
from allennlp.data.vocabulary import Vocabulary

from sherlock.feature_converters.feature_converter import FeatureConverter


class FeatureConverterAllennlp(FeatureConverter):
    """
    Abstract class for FeatureConverters to convert Documents into
    representation usable for allennlp.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexer: TokenIndexer,
        vocab: Vocabulary,
        labels: List[str],
        max_length: int = 512,
    ) -> None:
        super().__init__(labels, max_length)
        self.tokenizer = tokenizer
        self.token_indexer = token_indexer
        self.vocab = vocab

    @staticmethod
    def from_pretrained(
        path: str,
        tokenizer: Tokenizer,
        token_indexer: TokenIndexer
    ) -> "FeatureConverter":
        # TODO: Alternatively it would be better to save the token_indexer and
        # tokenizer name in the config, then load it here
        vocab_file = os.path.join(path, "converter_label_vocab.txt")
        converter_config_file = os.path.join(path, "converter_config.json")
        with open(converter_config_file, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        with open(vocab_file, "r", encoding="utf-8") as reader:
            config["labels"] = [line.strip() for line in reader.readlines()]
        config["tokenizer"] = tokenizer
        config["vocab"] = Vocabulary.from_files(str)
        config["token_indexer"] = token_indexer
        converter_class = FeatureConverter.by_name(config.pop("name"))
        return converter_class(**config)

    def save_vocabulary(self, vocab_path: str) -> None:
        """Save vocab and label_vocab"""
        self.vocab.save_to_files(vocab_path)
        super().save_vocabulary(vocab_path)
