import json
import os
from typing import Any, Dict, List, Set

from sherlock.dataset_readers.dataset_reader import DatasetReader
from sherlock.document import Document, Relation, Span, Token
from sherlock.tasks import IETask


class TacredDatasetReader(DatasetReader):
    """Dataset reader for the TACRED data set."""

    def __init__(
        self,
        data_dir: str,
        train_file: str = "train.json",
        dev_file: str = "dev.json",
        test_file: str = "test.json",
        negative_label_ner: str = "O",
        negative_label_re: str = "no_relation",
        convert_ptb_tokens: bool = True,
        tagging_scheme: str = "bio",
    ) -> None:
        super().__init__(data_dir)
        if tagging_scheme.lower() not in ["bio"]:
            raise ValueError("Unknown tagging scheme '%s'" % tagging_scheme)
        self.negative_label_ner = negative_label_ner
        self.negative_label_re = negative_label_re
        self.convert_ptb_tokens = convert_ptb_tokens
        self.tagging_scheme = tagging_scheme.lower()
        self.input_files = {
            split: os.path.join(data_dir, filename)
            for split, filename in zip(["train", "dev", "test"], [train_file, dev_file, test_file])
        }

    @staticmethod
    def _read_json(input_file: str) -> List[Dict[str, Any]]:
        with open(input_file, "r", encoding="utf-8") as tacred_file:
            data = json.load(tacred_file)
        return data

    def get_available_splits(self) -> List[str]:
        return ["train", "dev", "test"]

    def get_available_tasks(self) -> List[IETask]:
        return [IETask.NER, IETask.BINARY_RC]

    def get_documents(self, split: str) -> List[Document]:
        if split not in self.get_available_splits():
            raise ValueError("Selected split '%s' not available." % split)
        return self._create_documents(self._read_json(self.input_files[split]), split)

    def get_labels(self, task: IETask) -> List[str]:
        if task not in self.get_available_tasks():
            raise ValueError("Selected task '%s' not available." % task)

        dataset = self._read_json(self.input_files["train"])

        unique_labels: Set[str] = set()
        for example in dataset:
            if task == IETask.NER:
                ner = example["stanford_ner"] + [example["subj_type"], example["obj_type"]]
                unique_labels.update(ner)
            elif task == IETask.BINARY_RC:
                unique_labels.add(example["relation"])
            else:
                raise Exception("This should not happen.")

        labels = []
        if task == IETask.NER and self.tagging_scheme == "bio":
            # Make sure the negative label is always at position 0
            labels = [self.negative_label_ner]
            for label in unique_labels:
                if label != self.negative_label_ner:
                    labels.extend([prefix + label for prefix in ["B-", "I-"]])
        elif task == IETask.BINARY_RC:
            # Make sure the negative label is always at position 0
            labels = [self.negative_label_re]
            for label in unique_labels:
                if label not in labels:
                    labels.append(label)
        else:
            raise Exception("This should not happen.")

        return labels

    def get_additional_tokens(self, task: IETask) -> List[str]:
        additional_tokens: Set[str] = set()
        if task == IETask.BINARY_RC:
            dataset = self._read_json(self.input_files["train"])
            additional_tokens = set(["[HEAD_START]", "[HEAD_END]", "[TAIL_START]", "[TAIL_END]"])
            for example in dataset:
                head_type = "[HEAD=%s]" % example["subj_type"].upper()
                tail_type = "[TAIL=%s]" % example["obj_type"].upper()
                additional_tokens.add(head_type)
                additional_tokens.add(tail_type)

        return list(additional_tokens)

    def _create_documents(self, dataset: List[Dict[str, Any]], split: str):
        """Creates documents for the dataset."""
        return [self._example_to_document(example) for example in dataset]

    def _example_to_document(self, example: Dict[str, Any]) -> Document:
        tokens = example["token"]
        if self.convert_ptb_tokens:
            tokens = [self._convert_token(token) for token in tokens]

        ent_type = example["stanford_ner"]
        if ent_type and self.tagging_scheme == "bio":
            ent_type = self._ner_as_bio(example, insert_argument_types=True)

        pos = example["stanford_pos"]
        dep = example["stanford_deprel"]
        dep_head = example["stanford_head"]

        head_start, head_end = example["subj_start"], example["subj_end"] + 1
        tail_start, tail_end = example["obj_start"], example["obj_end"] + 1
        text = " ".join(tokens)

        doc = Document(guid=example["id"], text=text)

        start_offset = 0
        for idx, token in enumerate(tokens):
            end_offset = start_offset + len(token)
            doc.tokens.append(
                Token(
                    doc=doc,
                    start=start_offset,
                    end=end_offset,
                    lemma=token,
                    pos=pos[idx] if pos else None,
                    tag=pos[idx] if pos else None,
                    dep=dep[idx] if dep else None,
                    dep_head=dep_head[idx] if dep_head else None,
                    ent_type=ent_type[idx] if ent_type else None,
                )
            )
            # increment offset because of whitespace
            start_offset = end_offset + 1

        doc.sents = [Span(doc=doc, start=0, end=len(tokens))]

        # for label, (start, end) in bio_tags_to_spans(ner):
        #     # end is inclusive, we want exclusive -> +1
        #     doc.ments.append(Span(doc=doc, start=start, end=end + 1, label=label))

        doc.ments = [
            Span(doc=doc, start=head_start, end=head_end, label=example["subj_type"]),
            Span(doc=doc, start=tail_start, end=tail_end, label=example["obj_type"]),
        ]
        doc.rels = [Relation(doc=doc, head_idx=0, tail_idx=1, label=example["relation"])]
        return doc

    @staticmethod
    def _convert_token(token):
        """ Convert PTB tokens to normal tokens """
        return {
            "-lrb-": "(",
            "-rrb-": ")",
            "-lsb-": "[",
            "-rsb-": "]",
            "-lcb-": "{",
            "-rcb-": "}",
        }.get(token.lower(), token)

    @staticmethod
    def _ner_as_bio(example: Dict[str, Any], insert_argument_types: bool = False):
        tags = list(example["stanford_ner"])

        head_start, head_end = example["subj_start"], example["subj_end"] + 1
        tail_start, tail_end = example["obj_start"], example["obj_end"] + 1

        head_type = example["subj_type"]
        tail_type = example["obj_type"]

        if insert_argument_types:
            for i in range(head_start, head_end):
                tags[i] = head_type
            for i in range(tail_start, tail_end):
                tags[i] = tail_type

        bio_tags = []
        prev_tag = None
        for tag in tags:
            if tag == "O":
                bio_tags.append(tag)
                prev_tag = None
                continue
            prefix = "B-"
            if tag == prev_tag:
                prefix = "I-"
            bio_tags.append(prefix + tag)
            prev_tag = tag

        return bio_tags
