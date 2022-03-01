# -*- coding: utf8 -*-
"""

@date: 09.02.22
@author: gabriel.kressin@dfki.de

@description:
    Wrapper class implementing allennlp's own concept of
    the Dataset reader by using the sherlock DatasetReader and
    sherlock FeatureConverter.
"""
import logging
from typing import Iterable, Optional, Dict

from allennlp.data import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data import DatasetReader

from sherlock.dataset_readers import DatasetReader as DatasetReaderSherlock
from sherlock.feature_converters import FeatureConverter


logger = logging.getLogger(__name__)


@DatasetReader.register("sherlock")
class SherlockDatasetReader(DatasetReader):
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
    log_num_input_features : ``int``, optional (default=`None`)
        Amount of input features to be logged in logger.
    dataset_reader_kwargs : ``Dict[str, any]]``, optional (default=`None`)
        Additional keyword arguments for DatasetReader.
    feature_converter_kwargs : ``Dict[str, any]]``, optional (default=`None`)
        Additional keyword arguments for FeatureConverter.
    **kwargs : ``Dict[str, any]]``, optional (default=`None`)
        Additional kewyowrd arguments for allennlp DatasetReader class.
    """

    def __init__(
        self,
        dataset_reader_name: str,
        feature_converter_name: str,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        max_tokens: int=None,
        log_num_input_features: Optional[int]=None,
        dataset_reader_kwargs: Optional[Dict[str, any]]=None,
        feature_converter_kwargs: Optional[Dict[str, any]]=None,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)

        self.log_num_input_features = log_num_input_features or 0

        DatasetReaderClass = DatasetReaderSherlock.by_name(dataset_reader_name)
        self.dataset_reader: DatasetReaderSherlock = DatasetReaderClass(
            dataset_reader_kwargs
        )

        FeatureConverterClass = FeatureConverter.by_name(feature_converter_name)
        self.feature_converter: FeatureConverter = FeatureConverterClass(
            max_length=max_tokens,
            framework="allennlp",
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            **feature_converter_kwargs,
        )


    def _read(
        self,
        file_path: Optional[str]=None,
    ) -> Iterable[Instance]:

        # Get Document generator
        document_generator = self.dataset_reader.get_documents(file_path)

        for idx, document in enumerate(document_generator):
            # TODO: Make return type of document_to_features to iterable
            # (lazy loading -> performance boost)
            verbose = idx < self.log_num_input_features
            input_features = self.feature_converter.document_to_features(
                document, verbose
            )
            for input_feature in input_features:
                yield input_feature.instance
