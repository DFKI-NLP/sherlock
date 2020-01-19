from typing import List

from sherlock import Document


class DatasetReader:
    def __init__(self,
                 data_dir: str,
                 cache_features: bool = False) -> None:
        self.cache_features = cache_features

    def get_documents(self, split: str) -> List[Document]:
        raise NotImplementedError("DatasetReader must implement 'get_train_documents'")

    def get_labels(self) -> List[str]:
        raise NotImplementedError("DatasetReader must implement 'get_labels'")

    def get_additional_tokens(self) -> List[str]:
        raise NotImplementedError("DatasetReader must implement 'get_additional_tokens'")

    @property
    def available_splits(self) -> List[str]:
        raise NotImplementedError("DatasetReader must implement 'available_splits'")
