import os
import warnings
from typing import Any, Dict, List, Set, Optional, Iterable

from seqeval.metrics.sequence_labeling import get_entities

from sherlock.dataset_readers.dataset_reader import DatasetReader
from sherlock.document import Document, Mention, Span, Token
from sherlock.tasks import IETask


@DatasetReader.register("conll2003")
class Conll2003DatasetReader(DatasetReader):
    """
    Dataset reader for the CONLL_2003 dataset.

    Parameters
    ----------
    negative_label: ``str``, optional (default=`"O"`)
        label for instances in NER without a label class.
    **kwargs: ``Dict[str, Any]``, optional
        Catches keywords for backwards compability
        (`data_dir`, `train_file`, `dev_file`, `test_file`).
    """

    def __init__(
        self,
        negative_label: str="O",
        **kwargs,
    ) -> None:
        # for backward compability handle data_dir and train_file args
        # if data_dir is given, the get_documents and get_labels will return
        # lists instead of generators.
        data_dir = kwargs.get("data_dir")
        super().__init__(data_dir)

        if self.data_dir is not None:
            files = [
                kwargs.get("train_file") or "eng.train",
                kwargs.get("dev_file") or "eng.testa",
                kwargs.get("test_file") or "eng.testb",
            ]
            self.input_files = {
                split: os.path.join(data_dir, filename)
                for split, filename in zip(["train", "dev", "test"], files)
            }

        # Initialize other parameters
        self.negative_label = negative_label


    def get_documents(
        self,
        split: Optional[str]=None,
        file_path: str=None,
    ) -> Iterable[Document]:
        """
        Returns generator of Documents over data in file_path

        Parameters
        ----------
        file_path : ``str``
            path to data in json format
        split : ``str | None``, deprecated (default=``None``)
            only for backwards compability; in most cases ignored; do not use;
        """

        if split is not None and self.data_dir is not None:
            warnings.warn(
                "Using split as argument for get_documents is deprecated,"
                + " instead use `file_path`",
                DeprecationWarning,
            )
            if split not in self.get_available_splits():
                raise ValueError("Selected split '%s' not available." % split)

            # Returns list for backward compability
            return list(self._documents_generator(
                self._read_txt(split=split)))

        return self._documents_generator(
            self._read_txt(file_path=file_path))


    def get_labels(self, task: IETask, file_path: str=None) -> List[str]:
        if self.data_dir is not None:
            return list(self._labels_generator(task))
        elif file_path is None:
            raise AttributeError("get_labels requires file_path as argument")
        return list(self._labels_generator(task, file_path))


    def get_additional_tokens(self, task: IETask, file_path: str=None) -> List[str]:
        return []


    def get_available_splits(self) -> List[str]:
        warnings.warn(
            "`get_available_splits()` is deprecated.", DeprecationWarning)
        return ["train", "dev", "test"]


    def get_available_tasks(self) -> List[IETask]:
        return [IETask.NER]


    def _read_txt(
        self,
        split: str=None,
        file_path: str=None,
    ) -> List[Dict[str, Any]]:

        if self.data_dir is not None:
            file_path = self.input_files[split]
        else:
            split = os.path.basename(file_path)

        if file_path is None:
            raise AttributeError("bliblablup TODO")

        guid_index = 1
        dataset = []
        with open(file_path, encoding="utf-8") as conll_file:
            tokens = []  # type: List[str]
            ner = []  # type: List[str]
            for line in conll_file:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        dataset.append(
                            dict(
                                guid=f"{split}-{guid_index}",
                                tokens=tokens,
                                ner=ner
                            )
                        )
                        guid_index += 1
                        tokens = []
                        ner = []
                else:
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    if len(splits) > 1:
                        ner.append(splits[-1].replace("\n", ""))
                    # else:
                    # Examples could have no label for mode = "test"
                    # ner.append("O")
            if tokens:
                dataset.append(
                    dict(guid="{split}-{guid_index}", tokens=tokens, ner=ner))
        return dataset


    def _example_to_document(self, example: Dict[str, Any]) -> Document:
        tokens = example["tokens"]
        ner = example["ner"]
        text = " ".join(tokens)

        doc = Document(guid=example["guid"], text=text)
        doc.sents = [Span(doc=doc, start=0, end=len(tokens))]

        start_offset = 0
        for idx, token in enumerate(tokens):
            end_offset = start_offset + len(token)
            doc.tokens.append(
                Token(doc=doc, start=start_offset, end=end_offset, lemma=token, ent_type=ner[idx])
            )
            # increment offset because of whitespace
            start_offset = end_offset + 1

        for label, start, end in get_entities(ner):
            # end is inclusive, we want exclusive -> +1
            doc.ments.append(Mention(doc=doc, start=start, end=end + 1, label=label))
        return doc


    def _documents_generator(self, dataset: List[Dict[str, Any]]) -> Iterable[Document]:
        """Creates documents for the dataset."""
        for example in dataset:
            yield self._example_to_document(example)


    def _labels_generator(self, task: IETask, file_path: str=None) -> Iterable[str]:
        if task not in self.get_available_tasks():
            raise ValueError("Selected task '%s' not available." % task)

        if self.data_dir is not None:
            dataset = self._read_txt(split="train")
        else:
            dataset = self._read_txt(file_path=file_path)

        unique_labels: Set[str] = set()
        for example in dataset:
            unique_labels.update(example["ner"])

        # Make sure the negative label is always at position 0
        labels = [self.negative_label]
        yield self.negative_label
        for label in unique_labels:
            # Not sure why unique labels are tracked, but keep tracking
            if label not in labels:
                labels.append(label)
                yield label
