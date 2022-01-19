"""
Wrapper class implementing allennlp's own concept of
the Dataset reader.

Accomplished through the sherlock DatasetReader and
FeatureConverter
"""
import logging
from typing import Iterable, Optional, Dict

from allennlp.data import Instance
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data import DatasetReader

import sherlock
from sherlock.tasks import IETask
from sherlock.feature_converters import FeatureConverter
from sherlock.dataset_readers import DatasetReader as DatasetReaderSherlock


logger = logging.getLogger(__name__)


@DatasetReader.register("sherlock_reader")
class DatasetReaderAllennlp(DatasetReader):
    """
    Allennlp DatasetReader. Is realized by using a sherlock DatasetReader
    and sherlock FeatureConverter.

    Important: the sherlock FeatureConverter is only initialized AFTER calling
    self.read() for the first time (see below for reason)

    Note: this class does not have the "text_to_instance" function,
          because sherlock DatasetReaders do not produce text, but
          Documents. These are then converted with a FeatureConverter
          into allennlp Instances.

    Parameters
    ----------
    task : ``"ner" | "binary_rc"``
        Name of task for which data is loaded.
    dataset_reader_name : ``str``
        Unique identifier for sherlock DatasetReader (e.g. "tacred").
    feature_converter_name : ``str``
        Unique identifier for sherlock FeatureConverter (e.g. "binary_rc").
    tokenizer : ``Tokenizer``
        allennlp Tokenizer that can process Instances.
    token_indexer : ``TokenIndexer``
        allennlp TokenIndexer to index Instances.
    max_tokens : ``int``, optional (default=`None`)
        If set to a number, will limit sequences of tokens to maximum length.
    dataset_reader_kwargs : ``Dict[str, any]]``, optional (default=`None`)
        Additional keyword arguments for DatasetReader.
    feature_converter_kwargs : ``Dict[str, any]]``, optional (default=`None`)
        Additional keyword arguments for FeatureConverter.
    **kwargs : ``Dict[str, any]]``, optional (default=`None`)
        Additional kewyowrd arguments for allennlp DatasetReader class.
    """

    def __init__(
        self,
        task: str,
        dataset_reader_name: str,
        feature_converter_name: str,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        max_tokens: int=None,
        dataset_reader_kwargs: Optional[Dict[str, any]]=None,
        feature_converter_kwargs: Optional[Dict[str, any]]=None,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)

        # TODO: make this clean: is it enough to just take the string???
        if task == "binary_rc":
            self.task = IETask.BINARY_RC
        elif task == "ner":
            self.task = IETask.NER
        else:
            raise NotImplementedError("Task not implemented")

        DatasetReaderClass = DatasetReaderSherlock.by_name(dataset_reader_name)
        self.dataset_reader: DatasetReaderSherlock = DatasetReaderClass(
            dataset_reader_kwargs)

        # can only initialize FeatureConverter with labels, but labels are only
        # retrievable given data. Thus, initialize FeatureConverter later
        self.feature_converter_name: str = feature_converter_name
        self.feature_converter: Optional[FeatureConverter] = None

        self.feature_converter_kwargs = feature_converter_kwargs or {}
        self.feature_converter_kwargs["tokenizer"] = tokenizer
        self.feature_converter_kwargs["token_indexers"] = token_indexers
        self.feature_converter_kwargs["max_length"] = max_tokens


    def _read(
        self,
        file_path: Optional[str]=None,
    ) -> Iterable[Instance]:

        # Initialize FeatureConverter if that did not happen yet
        # Use opportunity to expand special tokens from tokenizer
        if self.feature_converter is None:

            # Get class
            FeatureConverterClass = \
                FeatureConverter.by_name(
                    self.feature_converter_name)
            # Initialize
            self.feature_converter = \
                FeatureConverterClass(
                    labels=self.dataset_reader.get_labels(self.task, file_path),
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
