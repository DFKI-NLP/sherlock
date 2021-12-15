import warnings
from typing import List, Iterable

from registrable import Registrable

from sherlock import Document
from sherlock.tasks import IETask


class DatasetReader(Registrable):
    """
    Superclass for every Dataset Reader in sherlock

    Parameters
    ----------
    data_dir: str, deprecated, optional (default=None)
        Directory in which data is saved
    """
    def __init__(self, data_dir: str=None) -> None:
        if data_dir is not None:
            warnings.warn(
                "Using data_dir for DatasetReader is deprecated, instead use"
                + " `file_path` argument in `get_documents`",
                DeprecationWarning
            )
        self.data_dir = data_dir

    def get_documents(self, file_path: str) -> Iterable[Document]:
        """
        Returns documents from data in given file_path

        Parameters
        ----------
        file_path : ``str``
            path to file containing data
        """

        raise NotImplementedError("DatasetReader must implement 'get_documents'")

    def get_labels(self, task: IETask, file_path: str) -> Iterable[str]:
        """
        Returns labels from data in given file_path for a task.

        Parameters
        ----------
        task : ``sherlock.task.IETask``
            task for which current DatasetReader is implemented
        file_path : ``str``
            path to file containing data
        """

        raise NotImplementedError("DatasetReader must implement 'get_labels'")

    def get_additional_tokens(self, task: IETask, file_path: str) -> List[str]:
        """
        Returns additional tokens from data in given file_path for a task.

        Parameters
        ----------
        task : ``sherlock.task.IETask``
            task for which current DatasetReader is implemented
        file_path : ``str``
            path to file containing data
        """

        raise NotImplementedError("DatasetReader must implement 'get_additional_tokens'")

    def get_available_splits(self) -> List[str]:
        warnings.warn("Using splits is deprecated", DeprecationWarning)
        raise NotImplementedError("DatasetReader must implement 'get_available_splits'")

    def get_available_tasks(self) -> List[IETask]:
        raise NotImplementedError("DatasetReader must implement 'get_available_tasks'")
