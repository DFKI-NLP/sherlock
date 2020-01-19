from typing import List

from sherlock import Document


class DatasetReader:
    def __init__(self,
                 data_dir: str) -> None:
        self.data_dir = data_dir

    def get_documents(self, split: str) -> List[Document]:
        raise NotImplementedError("DatasetReader must implement 'get_train_documents'")

    def get_labels(self) -> List[str]:
        raise NotImplementedError("DatasetReader must implement 'get_labels'")

    def get_additional_tokens(self) -> List[str]:
        raise NotImplementedError("DatasetReader must implement 'get_additional_tokens'")

    def get_available_splits(self) -> List[str]:
        raise NotImplementedError("DatasetReader must implement 'get_available_splits'")
