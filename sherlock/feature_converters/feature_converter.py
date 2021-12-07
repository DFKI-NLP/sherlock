import copy
import json
import logging
import os
from typing import List, Optional, Union

from registrable import Registrable

from sherlock import Document
from sherlock.feature_converters.input_features import InputFeatures


logger = logging.getLogger(__name__)

class FeatureConverter(Registrable):
    """
    Converts Document into Representation usable for Model Training.
    """

    def __init__(
        self, labels: List[str], max_length: int = 512
    ) -> None:
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
    def from_pretrained(path: str, *args, **kwargs) -> "FeatureConverter":
        return NotImplementedError("FeatureConverter must implement 'from_pretrained'.")

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
