"""
Wrapper class implementing allennlp's own concept of
the Dataset reader.

Accomplished through the sherlock DatasetReader and
FeatureConverter
"""
from typing import Iterable, Optional, Dict

from allennlp.data import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data import DatasetReader

import sherlock
from sherlock.tasks import IETask


@DatasetReader.register("sherlock_reader")
class DatasetReaderAllennlp(DatasetReader):

    def __init__(
        self,
        task: str,
        dataset_reader_name: str,
        feature_converter_name: str,
        tokenizer: Tokenizer=None,
        token_indexer: Dict[str, TokenIndexer]=None,
        max_tokens: int=None,
        feature_converter_kwargs: Optional[Dict[str, any]]=None,
        **kwargs
    ) -> None:
        """Initializes allennlp DatasetReader.

        Takes a sherlock DatasetReader and FeatureConverter and uses
        both of them to create the correct allennlp Instances.
        """
        super().__init__(**kwargs)

        # TODO: make this clean: is it enough to just take the string???
        if task == "binary_rc":
            self.task = IETask.BINARY_RC
        elif task == "ner":
            self.task = IETask.NER
        else:
            raise NotImplementedError("Task not implemented")

        DatasetReaderClass = \
            sherlock.dataset_readers.DatasetReader.by_name(dataset_reader_name)
        self.dataset_reader: sherlock.dataset_readers.DatasetReader = \
            DatasetReaderClass()

        # can only initialize FeatureConverter with labels, but labels are only
        # retrievable given data. Thus, initialize FeatureConverter later
        self.feature_converter_name = feature_converter_name
        self.feature_converter = None

        if feature_converter_kwargs is None:
            self.feature_converter_kwargs = {}
        else:
            self.feature_converter_kwargs = feature_converter_kwargs

        self.feature_converter_kwargs["tokenizer"] = tokenizer
        self.feature_converter_kwargs["token_indexer"] = token_indexer
        self.feature_converter_kwargs["max_length"] = max_tokens


    def _read(
        self,
        file_path: Optional[str]=None,
    ) -> Iterable[Instance]:
        """Returns iterable of allennlp instances."""

        # Initialize FeatureConverter if that did not happen yet
        if self.feature_converter is None:
            # Get class
            FeatureConverterClass = \
                sherlock.feature_converters.FeatureConverter.by_name(
                    self.feature_converter_name)
            # Initialize
            self.feature_converter = \
                FeatureConverterClass(
                    labels=self.dataset_reader.get_labels(self.task),
                    framework="allennlp",
                    **self.feature_converter_kwargs
                )

        # Get Document generator
        document_generator = self.dataset_reader.get_documents(file_path)

        for document in document_generator:
            # TODO: Make return type of document_to_features to iterable
            # (lazy loading -> performance boost)
            input_features = self.feature_converter.document_to_features(document)
            for input_feature in input_features:
                yield input_feature.instance
