from typing import List, Dict, Any

import os
import json
from collections import Counter

from sherlock.dataset_readers import DatasetReader
from sherlock.document import Token, Span, Relation, Document


class TacredDatasetReader(DatasetReader):
    """Dataset reader for the TACRED data set."""
    def __init__(self,
                 data_dir: str,
                 train_file: str = "train.json",
                 dev_file: str = "dev.json",
                 test_file: str = "test.json",
                 negative_label: str = "no_relation",
                 convert_ptb_tokens: bool = True) -> None:
        super().__init__(data_dir)
        self.convert_ptb_tokens = convert_ptb_tokens
        self.negative_label = negative_label
        self.input_files = {split: os.path.join(data_dir, filename)
                            for split, filename in zip(["train", "dev", "test"],
                                                       [train_file, dev_file, test_file])}

    @classmethod
    def _read_json(cls, input_file: str):
        with open(input_file, "r", encoding="utf-8") as tacred_file:
            data = json.load(tacred_file)
        return data

    @staticmethod
    def convert_token(token):
        """ Convert PTB tokens to normal tokens """
        if (token.lower() == '-lrb-'):
            return '('
        elif (token.lower() == '-rrb-'):
            return ')'
        elif (token.lower() == '-lsb-'):
            return '['
        elif (token.lower() == '-rsb-'):
            return ']'
        elif (token.lower() == '-lcb-'):
            return '{'
        elif (token.lower() == '-rcb-'):
            return '}'
        return token

    def get_available_splits(self) -> List[str]:
        return ["train", "dev", "test"]

    def get_documents(self, split: str) -> List[Document]:
        if split not in self.get_available_splits():
            raise ValueError("Selected split '%s' not available." % split)
        return self._create_documents(self._read_json(self.input_files[split]), split)

    def get_labels(self):
        dataset = self._read_json(self.input_files["train"])
        count = Counter()
        for example in dataset:
            count[example["relation"]] += 1
        # Make sure the negative label is always at position 0
        labels = [self.negative_label]
        for label, count in count.most_common():
            if label not in labels:
                labels.append(label)
        return labels

    def get_additional_tokens(self) -> List[str]:
        dataset = self._read_json(self.input_files["train"])
        additional_tokens = set(["[HEAD_START]", "[HEAD_END]",
                                 "[TAIL_START]", "[TAIL_END]"])
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
            tokens = [self.convert_token(token) for token in tokens]

        ner = example["stanford_ner"]
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
            doc.tokens.append(Token(doc=doc,
                                    start=start_offset,
                                    end=end_offset,
                                    lemma=token,
                                    pos=pos[idx],
                                    tag=pos[idx],
                                    dep=dep[idx],
                                    dep_head=dep_head[idx],
                                    ent_type=ner[idx]))
            # increment offset because of whitespace
            start_offset = end_offset + 1

        doc.sents = [Span(doc=doc, start=0, end=len(tokens))]
        doc.ments = [Span(doc=doc, start=head_start, end=head_end, label=example["subj_type"]),
                     Span(doc=doc, start=tail_start, end=tail_end, label=example["obj_type"])]
        doc.rels = [Relation(doc=doc,
                             head_idx=0,
                             tail_idx=1,
                             label=example["relation"])]
        return doc
