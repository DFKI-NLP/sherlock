# -*- coding: utf8 -*-
"""

@date: 09.02.22
@author: christoph.alt@posteo.de, gabriel.kressin@dfki.de, leonhard.hennig@dfki.de
"""
import json
import logging
import os
from typing import List, Optional, Dict, Union, Iterable

from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.token_indexers import TokenIndexer
from registrable import Registrable
from transformers import PreTrainedTokenizer

from sherlock import Document
from sherlock.feature_converters.input_features import (
    InputFeatures, InputFeaturesAllennlp, InputFeaturesTransformers)

logger = logging.getLogger(__name__)

# TODO: enable generator support for FeatureConverter
class FeatureConverter(Registrable):
    """
    Converts Document into Representation usable for Model Training.

    Parameters
    ----------
    labels : ``List[str]``
        All possible labels for a task
    max_length : ``int``, optional (default=`None`)
        If set to a number, will limit sequences to maximum length.
    framework : ``str``, optional (default=`transformers`)
        Whether to use `transformers` or `allennlp`
    **kwargs : ``Dict[str,any]``
        init arguments for `transformer` or `allennlp` FeatureConverter:
        `transformer`:  {"tokenizer": PreTrainedTokenizer}
        `allennlp`:
            {
                "tokenizer": Tokenizer,
                "token_indexers": Dict[str, TokenIndexer]
            }
    """

    def __init__(
        self,
        labels: Iterable[str],
        max_length: Optional[int] =None,
        framework: str ="transformers",
        **kwargs,
    ) -> None:
        self.labels = list(labels)
        self.max_length = max_length
        self.id_to_label_map = {i: l for i, l in enumerate(self.labels)}
        self.label_to_id_map = {l: i for i, l in enumerate(self.labels)}
        self.framework = framework

        if framework == "transformers":
            logger.info("Initializing Transformers FeatureConverter")
            self._init_feature_converter_transformers(
                **{k: v for k, v in kwargs.items() if k in ["tokenizer"]}
            )
        elif framework == "allennlp":
            logger.info("Initializing AllenNLP FeatureConverter")
            self._init_feature_converter_allennlp(**{k: v for k, v in kwargs.items()
                                                     if k in ["tokenizer", "token_indexers"]})
        else:
            raise NotImplementedError(f"Framework not supported: {framework}")

    def _init_feature_converter_transformers(
        self, tokenizer: PreTrainedTokenizer
    ) -> None:
        self.tokenizer = tokenizer

    def _init_feature_converter_allennlp(
        self, tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer]
    ) -> None:
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

    @property
    def name(self) -> str:
        raise NotImplementedError("FeatureConverter must implement 'name'.")

    @property
    def persist_attributes(self) -> List[str]:
        """All attributes that are saved alongside FeatureConverter"""
        raise NotImplementedError("FeatureConverter must implement 'persist_attributes'.")

    def document_to_features_transformers(
        self, document: Document, verbose: bool=False
    ) -> List[InputFeaturesTransformers]:
        raise NotImplementedError(
            "FeatureConverter does not implement 'document_to_features_transformers'.")

    def document_to_features_allennlp(
        self, document: Document, verbose: bool=False
    ) -> List[InputFeaturesAllennlp]:
        raise NotImplementedError(
            "FeatureConverter does not implement 'document_to_features_allennlp'.")

    def document_to_features(
        self, document: Document, verbose: bool=False
    ) -> List[InputFeatures]:
        if self.framework == "transformers":
            return self.document_to_features_transformers(document, verbose)
        elif self.framework == "allennlp":
            return self.document_to_features_allennlp(document, verbose)

    def documents_to_features(
        self, documents: List[Document], verbose: bool=False
    ) -> List[List[InputFeatures]]:
        input_features = []
        for document in documents:
            input_features.extend(self.document_to_features(document, verbose))
        return input_features

    @staticmethod
    def _from_pretrained_transformers(
        path: str, config: Dict[str,any], tokenizer: PreTrainedTokenizer
    ) -> "FeatureConverter":
        vocab_file = os.path.join(path, "converter_label_vocab.txt")
        with open(vocab_file, "r", encoding="utf-8") as reader:
            config["labels"] = [line.strip() for line in reader.readlines()]
        config["tokenizer"] = tokenizer
        converter_class = FeatureConverter.by_name(config.pop("name"))
        return converter_class(**config)

    @staticmethod
    def _from_pretrained_allennlp(
        path: str,
        config: Dict[str,any],
        tokenizer: Tokenizer,
        token_indexers: Dict[str,TokenIndexer],
    ) -> "FeatureConverter":
        # TODO: Alternatively it would be better to save the token_indexers and
        # tokenizer name in the config, then load it here
        vocab_file = os.path.join(path, "converter_label_vocab.txt")
        with open(vocab_file, "r", encoding="utf-8") as reader:
            config["labels"] = [line.strip() for line in reader.readlines()]
        config["tokenizer"] = tokenizer
        config["token_indexers"] = token_indexers
        converter_class = FeatureConverter.by_name(config.pop("name"))
        return converter_class(**config)


    @staticmethod
    def from_pretrained(path: str, **kwargs) -> "FeatureConverter":
        """Load FeatureConverter from a directory it was previously saved in"""
        # 1. get config and framework
        converter_config_file = os.path.join(path, "converter_config.json")
        with open(converter_config_file, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        framework = config.get("framework")
        framework = "transformers" if framework is None else framework # backwards compability

        # 2. Call framework specific constructor
        if framework == "transformers":
            return FeatureConverter._from_pretrained_transformers(path, config, **kwargs)
        elif framework == "allennlp":
            return FeatureConverter._from_pretrained_allennlp(path, config, **kwargs)

    def save_vocabulary(self, vocab_path: str) -> None:
        raise DeprecationWarning("Deprecated. Use 'save_label_vocabulary'.")

    def save_label_vocabulary(self, vocab_path: str) -> None:
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
                        f"Saving label vocabulary to {vocab_file}: vocabulary"
                        + " indices are not consecutive. Please check that"
                        + " the vocabulary is not corrupted!",
                    )
                    index = label_index
                writer.write(label + "\n")
                index += 1

    def save(self, save_directory: str) -> None:
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
        self.save_label_vocabulary(save_directory)
        config = dict(
            name=self.name,
            framework=self.framework,
            **{attr: getattr(self, attr) for attr in self.persist_attributes}
        )
        converter_config_file = os.path.join(save_directory, "converter_config.json")
        with open(converter_config_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(config, ensure_ascii=False))

    @staticmethod
    def _log_input_features_transformers(
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

    @staticmethod
    def _log_input_features_allennlp(
            tokens: List[Token],
            document: Document,
            features: InputFeatures,
            labels: Optional[Union[str, List[str]]] = None,
    ) -> None:
        # Examples are unpadded and not "officially" tokenized by allennlp yet
        logger.info("*** Example ***")
        logger.info(f"guid: {document.guid}", )
        logger.info(f"tokens: {' '.join([x.text for x in tokens])}")
        if len(tokens) > 0:
            if tokens[0].text_id is not None:
                logger.info(f"token_ids: {' '.join([str(x.text_id) for x in tokens])}")
            if tokens[0].type_id is not None:
                logger.info(f"token_type_ids: {' '.join([str(x.type_id) for x in tokens])}")
        if labels:
            logger.info(f"labels: {labels}")
        logger.info(features.instance)

    def _log_input_features(
        self,
        *args,
        **kwargs,
    ) -> None:
        if self.framework == "transformers":
            self._log_input_features_transformers(*args, **kwargs)
        elif self.framework == "allennlp":
            self._log_input_features_allennlp(*args, **kwargs)
