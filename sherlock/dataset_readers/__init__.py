from .conll_2003 import Conll2003DatasetReader
from .dataset_reader import DatasetReader
from .tacred import TacredDatasetReader
from .dataset_reader_allennlp import DatasetReaderAllennlp


__all__ = ["DatasetReader", "TacredDatasetReader", "Conll2003DatasetReader",
    "DatasetReaderAllennlp"]
