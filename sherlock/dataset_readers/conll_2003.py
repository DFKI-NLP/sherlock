import os
from typing import Any, Dict, List, Set

from seqeval.metrics.sequence_labeling import get_entities

from sherlock.dataset_readers.dataset_reader import DatasetReader
from sherlock.document import Document, Mention, Span, Token
from sherlock.tasks import IETask

@DatasetReader.register("conll2003")
class Conll2003DatasetReader(DatasetReader):
    def __init__(
        self,
        data_dir: str,
        train_file: str = "eng.train",
        dev_file: str = "eng.testa",
        test_file: str = "eng.testb",
        negative_label: str = "O",
    ) -> None:
        super().__init__(data_dir)
        self.negative_label = negative_label
        self.input_files = {
            split: os.path.join(data_dir, filename)
            for split, filename in zip(["train", "dev", "test"], [train_file, dev_file, test_file])
        }

    def _read_txt(self, split: str) -> List[Dict[str, Any]]:
        guid_index = 1
        dataset = []
        with open(self.input_files[split], encoding="utf-8") as conll_file:
            tokens = []  # type: List[str]
            ner = []  # type: List[str]
            for line in conll_file:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        dataset.append(
                            dict(guid="{}-{}".format(split, guid_index), tokens=tokens, ner=ner)
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
                dataset.append(dict(guid="{}-{}".format(split, guid_index), tokens=tokens, ner=ner))
        return dataset

    def get_available_splits(self) -> List[str]:
        return ["train", "dev", "test"]

    def get_available_tasks(self) -> List[IETask]:
        return [IETask.NER]

    def get_documents(self, split: str) -> List[Document]:
        if split not in self.get_available_splits():
            raise ValueError("Selected split '%s' not available." % split)
        return self._create_documents(self._read_txt(split), split)

    def get_labels(self, task: IETask) -> List[str]:
        if task not in self.get_available_tasks():
            raise ValueError("Selected task '%s' not available." % task)

        dataset = self._read_txt(split="train")

        unique_labels: Set[str] = set()
        for example in dataset:
            unique_labels.update(example["ner"])

        # Make sure the negative label is always at position 0
        labels = [self.negative_label]
        for label in unique_labels:
            if label not in labels:
                labels.append(label)
        return labels

    def get_additional_tokens(self, task: IETask) -> List[str]:
        return []

    def _create_documents(self, dataset: List[Dict[str, Any]], split: str):
        """Creates documents for the dataset."""
        return [self._example_to_document(example) for example in dataset]

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
