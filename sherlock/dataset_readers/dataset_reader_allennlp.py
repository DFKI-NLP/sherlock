"""
Wrapper class to create a class after allennlps own concept of
the Dataset reader
"""
from typing import Iterable, List, Optional, Dict

from allennlp import data
from allennlp.data import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer

import sherlock
from sherlock.dataset_readers.dataset_reader import DatasetReader
from sherlock.document import Document
from sherlock.tasks import IETask


class DatasetReaderAllennlp(data.DatasetReader):

    def __init__(
        self,
        task: str,
        dataset_reader_name: str,
        feature_converter_name: str,
        tokenizer: Tokenizer=None,
        token_indexers: Dict[str, TokenIndexer]=None,
        max_tokens: int=None,
        feature_converter_kwargs: Optional[Dict[str, any]]=None,
        **kwargs
    ) -> None:
        """Initializes allennlp DatasetReader.

        Takes a sherlock DatasetReader and FeatureConverter and uses
        both of them to create the correct allennlp Instances.
        """
        super().__init__(**kwargs)

        # TODO: make this clean
        if task == "binary_rc":
            self.task = IETask.BINARY_RC
        elif task == "ner":
            self.task = IETask.NER
        else:
            raise NotImplementedError("Task not implemented")

        DatasetReaderClass = \
            sherlock.dataset_readers.DatasetReader.by_name(dataset_reader_name)
        # set data_dir later in order to adhere to allennlp DatasetReader api
        self.dataset_reader: sherlock.dataset_readers.DatasetReader = \
            DatasetReaderClass(data_dir="")

        # can only initialize FeatureConverter with labels, but labels only
        # are retrievable with data_dir. Thus, initialize FeatureConverter later
        self.feature_converter_name = feature_converter_name
        self.feature_converter = None

        if feature_converter_kwargs is None:
            self.feature_converter_kwargs = {}
        else:
            self.feature_converter_kwargs = feature_converter_kwargs

        self.feature_converter_kwargs["tokenizer"] = tokenizer
        self.feature_converter_kwargs["token_indexers"] = token_indexers
        self.feature_converter_kwargs["max_length"] = max_tokens


    def _read(self, file_path: Optional[str]=None) -> Iterable[Instance]:
        """Returns iterable of allennlp instances.

        If sherlock DatasetReader has been initialized with
        the correct data_dir no file_path is required.
        TODO: this does not feel clean.
        """
        # 1. Set DatasetReader dir
        if file_path is not None:
            self.dataset_reader.data_dir = file_path

        # 2. Init FeatureConverter if that did not happen yet
        if self.feature_converter is None:

            labels = self.dataset_reader.get_labels(self.task)
            FeatureConverterClass = \
                sherlock.feature_converters.FeatureConverter.by_name(
                    self.feature_converter_name)
            self.feature_converter = \
                FeatureConverterClass(
                    labels,
                    framework="allennlp",
                    **self.feature_converter_kwargs
                )

        # 3. Get Documents
        documents: List[Document] = self.dataset_reader.get_documents()

        # 4. Convert Documents to Instances
        for document in documents:
            input_features = self.feature_converter.document_to_features(document)
            for input_feature in input_features:
                yield input_feature.instance
