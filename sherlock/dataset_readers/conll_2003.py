from typing import List, Dict, Any

import os
from collections import Counter

from sherlock.document import Token, Span, Document
from sherlock.dataset_readers.span_utils import bio_tags_to_spans


class Conll2003DatasetReader:
    def __init__(self,
                 data_dir: str,
                 train_file: str = "eng.train",
                 dev_file: str = "eng.testa",
                 test_file: str = "eng.testb",
                 negative_label: str = "O") -> None:
        self.data_dir = data_dir
        self.negative_label = negative_label
        self.input_files = {split: os.path.join(data_dir, filename)
                            for split, filename in zip(["train", "dev", "test"],
                                                       [train_file, dev_file, test_file])}

    def _read_txt(self, split: str) -> List[Dict[str, Any]]:
        guid_index = 1
        dataset = []
        with open(self.input_files[split], encoding="utf-8") as conll_file:
            tokens = []  # type: List[str]
            labels = []  # type: List[str]
            for line in conll_file:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        dataset.append(dict(guid="{}-{}".format(split, guid_index),
                                            tokens=tokens,
                                            labels=labels))
                        guid_index += 1
                        tokens = []
                        labels = []
                else:
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    # else:
                        # Examples could have no label for mode = "test"
                        # labels.append("O")
            if tokens:
                dataset.append(dict(guid="{}-{}".format(split, guid_index),
                                    tokens=tokens,
                                    labels=labels))
        return dataset

    def get_available_splits(self) -> List[str]:
        return ["train", "dev", "test"]

    def get_documents(self, split: str) -> List[Document]:
        if split not in self.get_available_splits():
            raise ValueError("Selected split '%s' not available." % split)
        return self._create_documents(self._read_txt(split), split)

    def get_labels(self):
        dataset = self._read_txt(split="train")
        count = Counter()
        for example in dataset:
            for label in example["labels"]:
                count[label] += 1
        # Make sure the negative label is always at position 0
        labels = [self.negative_label]
        for label, count in count.most_common():
            if label not in labels:
                labels.append(label)
        return labels

    def get_additional_tokens(self) -> List[str]:
        return []

    def _create_documents(self, dataset: List[Dict[str, Any]], split: str):
        """Creates documents for the dataset."""
        return [self._example_to_document(example) for example in dataset]

    def _example_to_document(self, example: Dict[str, Any]) -> Document:
        tokens = example["tokens"]
        labels = example["labels"]
        text = " ".join(tokens)

        doc = Document(guid=example["guid"], text=text)

        start_offset = 0
        for token, label in zip(tokens, labels):
            end_offset = start_offset + len(token)
            doc.tokens.append(Token(doc=doc,
                                    start=start_offset,
                                    end=end_offset,
                                    lemma=token,
                                    ent_type=label))
            # increment offset because of whitespace
            start_offset = end_offset + 1

        doc.sents = [Span(doc=doc, start=0, end=len(tokens))]

        for label, (start, end) in bio_tags_to_spans(labels):
            # end is inclusive, we want exclusive -> +1
            doc.ments.append(Span(doc=doc, start=start, end=end + 1, label=label))

        return doc
