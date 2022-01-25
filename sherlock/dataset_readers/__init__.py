from .conll_2003 import Conll2003DatasetReader
from .dataset_reader import DatasetReader
from .tacred import TacredDatasetReader


__all__ = ["DatasetReader", "TacredDatasetReader", "Conll2003DatasetReader",
    "DatasetReaderAllennlp"]
