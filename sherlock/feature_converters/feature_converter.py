import copy
import json
import logging
import os
from typing import List, Optional, Union

from registrable import Registrable
from transformers import PreTrainedTokenizer

from sherlock import Document


logger = logging.getLogger(__name__)


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        metadata=None,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.head_mask = head_mask
        self.labels = labels
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return self.to_dict()

    def __repr__(self) -> str:
        return str(self.to_dict())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class FeatureConverter(Registrable):
    """
    Converts Document into Representation usable for Model Training.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, labels: List[str], max_length: int = 512
    ) -> None:
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        self.id_to_label_map = {i: l for i, l in enumerate(labels)}
        self.label_to_id_map = {l: i for i, l in enumerate(labels)}

    @property
    def name(self) -> str:
        raise NotImplementedError("FeatureConvert must implement 'name'.")

    @property
    def persist_attributes(self) -> List[str]:
        raise NotImplementedError("FeatureConvert must implement 'persist_attributes'.")

    def document_to_features(
        self, document: Document, verbose: bool = False
    ) -> List[InputFeatures]:
        raise NotImplementedError("FeatureConvert must implement 'document_to_features'.")

    def documents_to_features(self, documents: List[Document]) -> List[InputFeatures]:
        input_features = []
        for document in documents:
            input_features.extend(self.document_to_features(document))
        return input_features

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

    def save(self, save_directory: str) -> None:
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
        self.save_vocabulary(save_directory)
        config = dict(
            name=self.name, **{attr: getattr(self, attr) for attr in self.persist_attributes}
        )
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
                        "Saving vocabulary to %s: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!",
                        vocab_file,
                    )
                    index = label_index
                writer.write(label + "\n")
                index += 1

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
