# -*- coding: utf8 -*-
"""

@date: 09.02.22
@author: christoph.alt@posteo.de, gabriel.kressin@dfki.de, leonhard.hennig@dfki.de
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Iterable
import warnings

from sherlock.dataset_readers.dataset_reader import DatasetReader
from sherlock.document import Document, Mention, Relation, Span, Token
from sherlock.tasks import IETask


INVERSE_RELATIONS = {
    "per:alternate_names": "per:alternate_names",
    "per:children": "per:parents",
    "per:parents": "per:children",
    "per:siblings": "per:siblings",
    "per:spouse": "per:spouse",
    "per:other_family": "per:other_family",
    "org:alternate_names": "org:alternate_names",
    "org:member_of": "org:members",
    "org:members": "org:member_of",
    "org:parents": "org:subsidiaries",
    "org:subsidiaries": "org:parents",
    # "org:top_members/employees": "per:employee_of",
    # "per:employee_of": "org:top_members/employees",
}

logger = logging.getLogger(__name__)


@DatasetReader.register("tacred")
class TacredDatasetReader(DatasetReader):
    """
    Dataset reader for the TACRED dataset.

    Parameters
    ----------
    negative_label_ner : ``str``, optional (default=`"O"`)
        label for instances in NER without a label class.
    negative_label_re : ``str``, optional (default=`"no_relation"`)
        label for instances without relation in Relation Extraction.
    convert_ptb_tokens: ``bool``, optional (default=`True`)
        flag to convert PTB tokens to normal tokens.
    tagging_scheme: ``str``, optional (default=`"bio"`)
        tagging scheme for charactes in sentences. Only supports `"bio"`.
    add_inverse_relations: ``bool``, optional (default=`False`)
        for any relation in TACRED, add the inversion as another instance.
    max_instances: ``int``, optional (default=`None`)
        Only use this number of first instances in dataset (e.g. for debugging).
    **kwargs: ``Dict[str, Any]``
        Catches keywords for backwards compability
        (`data_dir`, `train_file`, `dev_file`, `test_file`).
    """

    def __init__(
        self,
        negative_label_ner: str="O",
        negative_label_re: str="no_relation",
        convert_ptb_tokens: bool=True,
        tagging_scheme: str="bio",
        add_inverse_relations: bool=False,
        max_instances: Optional[int]=None,
        **kwargs,
    ) -> None:
        # for backward compability handle data_dir and train_file args
        # if data_dir is given, the get_documents and get_labels will return
        # lists instead of generators.
        data_dir = kwargs.get("data_dir")
        super().__init__(data_dir)

        if self.data_dir is not None:
            files = [
                kwargs.get("train_file") or "train.json",
                kwargs.get("dev_file") or "dev.json",
                kwargs.get("test_file") or "test.json",
            ]
            self.input_files = {
                split: os.path.join(data_dir, filename)
                for split, filename in zip(["train", "dev", "test"], files)
            }

        # initialize other parameters
        if tagging_scheme.lower() not in ["bio"]:
            raise ValueError("Unknown tagging scheme '%s'" % tagging_scheme)
        self.negative_label_ner = negative_label_ner
        self.negative_label_re = negative_label_re
        self.convert_ptb_tokens = convert_ptb_tokens
        self.tagging_scheme = tagging_scheme.lower()
        self.add_inverse_relations = add_inverse_relations
        self.max_instances = max_instances
        self.use_dfki_jsonl_format = kwargs.get("tacred_use_dfki_jsonl_format") or False


    def get_documents(
        self,
        file_path: str=None,
        split: Optional[str]=None
    ) -> Iterable[Document]:
        """
        Returns generator of Documents over data in file_path

        Parameters
        ----------
        file_path : ``str``
            path to data in json format.
        split : ``str | None``, deprecated (default=`None`)
            only for backwards compability; in most cases ignored; do not use;
        """

        if split is not None and self.data_dir is not None:
            warnings.warn(
                "Using split as argument for get_documents is deprecated,"
                + "instead use `file_path`",
                DeprecationWarning,
            )
            if split not in self.get_available_splits():
                raise ValueError("Selected split '%s' not available." % split)
            file_path = self.input_files[split]
            # Returns list for backward compability
            return list(self._documents_generator(self._read_json(file_path, self.use_dfki_jsonl_format)))

        # Returns generator for performance improvements
        return self._documents_generator(self._read_json(file_path, self.use_dfki_jsonl_format))


    def get_labels(self, task: IETask, file_path: str=None) -> List[str]:
        if self.data_dir is not None:
            return list(self._labels_generator(task))
        elif file_path is None:
            raise AttributeError("get_labels requires file_path as argument")
        return list(self._labels_generator(task, file_path))


    def get_additional_tokens(self, task: IETask, file_path: str=None) -> List[str]:
        additional_tokens: Set[str] = set()
        if task == IETask.BINARY_RC:
            # backwards compability
            if self.data_dir is not None:
                dataset = list(self._read_json(self.input_files["train"], self.use_dfki_jsonl_format))
            else:
                if file_path is None:
                    raise AttributeError(
                        "get_additional_tokens requires file_path as argument")
                dataset = self._read_json(file_path, self.use_dfki_jsonl_format)

            additional_tokens = \
                set(["[HEAD_START]", "[HEAD_END]", "[TAIL_START]", "[TAIL_END]"])
            for example in dataset:
                head_type = "[HEAD=%s]" % (example["type"][0] if self.use_dfki_jsonl_format else example["subj_type"].upper())
                tail_type = "[TAIL=%s]" % (example["type"][1] if self.use_dfki_jsonl_format else example["obj_type"].upper())
                additional_tokens.add(head_type)
                additional_tokens.add(tail_type)

        return sorted(list(additional_tokens))


    def get_available_splits(self) -> List[str]:
        warnings.warn(
            "`get_available_splits()` is deprecated.", DeprecationWarning)
        return ["train", "dev", "test"]


    def get_available_tasks(self) -> List[IETask]:
        return [IETask.NER, IETask.BINARY_RC]


    @staticmethod
    def _read_json(input_file: str, use_dfki_jsonl_format:bool) -> List[Dict[str, Any]]:
        with open(input_file, "r", encoding="utf-8") as tacred_file:
            if not use_dfki_jsonl_format:
                data = json.load(tacred_file)
            else:
                data = []
                for line in tacred_file:
                    data.append(json.loads(line))
        return data


    @staticmethod
    def _convert_token(token):
        """Convert PTB tokens to normal tokens"""
        return {
            "-lrb-": "(",
            "-rrb-": ")",
            "-lsb-": "[",
            "-rsb-": "]",
            "-lcb-": "{",
            "-rcb-": "}",
        }.get(token.lower(), token)


    def _example_to_document(self, example: Dict[str, Any]) -> Optional[Document]:
        tokens = example["tokens"] if self.use_dfki_jsonl_format else example["token"]
        if self.convert_ptb_tokens:
            tokens = [self._convert_token(token) for token in tokens]
        text = " ".join(tokens)

        head_start, head_end = (example["entities"][0][0], example["entities"][0][1]) if self.use_dfki_jsonl_format else (example["subj_start"], example["subj_end"] + 1)
        tail_start, tail_end = (example["entities"][1][0], example["entities"][1][1]) if self.use_dfki_jsonl_format else (example["obj_start"], example["obj_end"] + 1)

        if head_end > len(tokens) or tail_end > len(tokens):
            return None

        ent_type = example.get("stanford_ner")
        if ent_type and self.tagging_scheme == "bio":
            ent_type = self._ner_as_bio(example, insert_argument_types=True, use_dfki_jsonl_format=self.use_dfki_jsonl_format)

        pos = example.get("stanford_pos")
        dep = example.get("stanford_deprel")
        dep_head = example.get("stanford_head")

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
        subj_type = example["type"][0] if self.use_dfki_jsonl_format else example["subj_type"]
        obj_type = example["type"][1] if self.use_dfki_jsonl_format else example["obj_type"]
        doc.ments = [
            Mention(doc=doc, start=head_start, end=head_end, label=subj_type),
            Mention(doc=doc, start=tail_start, end=tail_end, label=obj_type),
        ]
        rel_label = example["label"] if self.use_dfki_jsonl_format else example["relation"]
        doc.rels = [Relation(doc=doc, head_idx=0, tail_idx=1, label=rel_label)]
        if self.add_inverse_relations:
            doc.rels.append(
                Relation(
                    doc=doc,
                    head_idx=1,
                    tail_idx=0,
                    label=INVERSE_RELATIONS.get(rel_label, self.negative_label_re),
                )
            )

        return doc


    def _documents_generator(self, dataset: List[Dict[str, Any]]) -> Iterable[Document]:
        read_instances = 0
        for example in dataset:
            document = self._example_to_document(example)
            if document is None:
                logger.info(f"Skipped document with id: {example['id']}")
                continue

            yield document
            read_instances += 1
            if (
                self.max_instances is not None
                and read_instances >= self.max_instances
            ):
                break


    def _labels_generator(self, task: IETask, file_path: str=None) -> Iterable[str]:
        if task not in self.get_available_tasks():
            raise ValueError("Selected task '%s' not available." % task)

        # Backwards compability
        if self.data_dir is not None:
            dataset = list(self._read_json(self.input_files["train"], self.use_dfki_jsonl_format))
        else:
            dataset = self._read_json(file_path, self.use_dfki_jsonl_format)

        unique_labels: Set[str] = set()
        for example in dataset:
            if task == IETask.NER:
                ner = example.get("stanford_ner", []) + example["type"] if self.use_dfki_jsonl_format else [example["subj_type"], example["obj_type"]]
                unique_labels.update(ner)
            elif task == IETask.BINARY_RC:
                unique_labels.add(example["label"] if self.use_dfki_jsonl_format else example["relation"])
            else:
                raise Exception("This should not happen.")


        if task == IETask.NER and self.tagging_scheme == "bio":
            # Make sure the negative label is always at position 0
            yield self.negative_label_ner
            for label in unique_labels:
                if label != self.negative_label_ner:
                    ret_labels = [prefix + label for prefix in ["B-", "I-"]]
                    for ret_label in ret_labels:
                        yield ret_label
        elif task == IETask.BINARY_RC:
            # Make sure the negative label is always at position 0
            labels = [self.negative_label_re]
            yield self.negative_label_re
            for label in unique_labels:
                # Not sure why unique labels are tracked, but keep tracking
                if label not in labels:
                    labels.append(label)
                    yield label
        else:
            raise Exception("This should not happen.")


    @staticmethod
    def _ner_as_bio(example: Dict[str, Any], insert_argument_types: bool = False, use_dfki_jsonl_format: bool = False):
        tags = list(example["stanford_ner"])

        head_start, head_end = (example["entities"][0][0], example["entities"][0][1]) if use_dfki_jsonl_format else (example["subj_start"], example["subj_end"] + 1)
        tail_start, tail_end = (example["entities"][1][0], example["entities"][1][1]) if use_dfki_jsonl_format else (example["obj_start"], example["obj_end"] + 1)

        head_type = example["type"][0] if use_dfki_jsonl_format else example["subj_type"]
        tail_type = example["type"][1] if use_dfki_jsonl_format else example["obj_type"]

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
