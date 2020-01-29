import os

from sherlock.dataset_readers import TacredDatasetReader
from sherlock.tasks import IETask
from tests import FIXTURES_ROOT


def test_get_documents():
    reader = TacredDatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"), train_file="tacred.json"
    )

    documents = reader.get_documents(split="train")
    assert len(documents) == 3

    doc = documents[1]
    assert len(doc.tokens) == 34
    assert len(doc.ments) == 2
    assert len(doc.rels) == 1
    assert doc.tokens[1].text == "District"

    relation = doc.rels[0]
    assert relation.head.start == 17
    assert relation.head.end == 19
    assert relation.tail.start == 4
    assert relation.tail.end == 6
    assert relation.label == "no_relation"


def test_get_labels_ner():
    reader = TacredDatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"), train_file="tacred.json"
    )

    expected_labels = ["O"]
    expected_labels += [
        prefix + label
        for prefix in ["B-", "I-"]
        for label in [
            "PERSON",
            "MISC",
            "TITLE",
            "LOCATION",
            "ORGANIZATION",
            "CITY",
            "TIME",
            "NUMBER",
        ]
    ]

    labels = reader.get_labels(IETask.NER)
    assert sorted(labels) == sorted(expected_labels)


def test_get_additional_tokens_ner():
    reader = TacredDatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"), train_file="tacred.json"
    )

    labels = reader.get_additional_tokens(IETask.NER)
    assert labels == []


def test_get_labels_re():
    reader = TacredDatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"), train_file="tacred.json"
    )

    labels = reader.get_labels(IETask.BINARY_RC)
    assert sorted(labels) == sorted(["no_relation", "per:title", "per:children"])


def test_get_additional_tokens_re():
    reader = TacredDatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"), train_file="tacred.json"
    )

    additional_tokens = reader.get_additional_tokens(IETask.BINARY_RC)

    assert sorted(additional_tokens) == sorted(
        [
            "[HEAD_START]",
            "[HEAD_END]",
            "[TAIL_START]",
            "[TAIL_END]",
            "[HEAD=PERSON]",
            "[TAIL=TITLE]",
            "[TAIL=PERSON]",
            "[TAIL=CITY]",
        ]
    )


def test_convert_ptb_token():
    not_convert_reader = TacredDatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
        train_file="tacred.json",
        convert_ptb_tokens=False,
    )
    convert_reader = TacredDatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
        train_file="tacred.json",
        convert_ptb_tokens=True,
    )

    example = {
        "id": "test",
        "docid": "test",
        "relation": "test",
        "token": ["-LRB-", "-RRB-", "-LSB-", "-RSB-", "-LCB-", "-RCB-"],
        "subj_start": 0,
        "subj_end": 0,
        "obj_start": 1,
        "obj_end": 1,
        "subj_type": "TEST",
        "obj_type": "TEST",
        "stanford_pos": [],
        "stanford_ner": [],
        "stanford_head": [],
        "stanford_deprel": [],
    }

    not_convert_doc = not_convert_reader._example_to_document(example)
    assert [t.text for t in not_convert_doc.tokens] == [
        "-LRB-",
        "-RRB-",
        "-LSB-",
        "-RSB-",
        "-LCB-",
        "-RCB-",
    ]

    convert_doc = convert_reader._example_to_document(example)
    assert [t.text for t in convert_doc.tokens] == ["(", ")", "[", "]", "{", "}"]


def test_add_inverse_relations():
    reader = TacredDatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
        train_file="tacred.json",
        add_inverse_relations=True,
    )

    documents = reader.get_documents(split="train")
    assert len(documents) == 3

    assert all([len(doc.rels) == 2 for doc in documents])

    assert [rel.label for rel in documents[0].rels] == ["per:title", "no_relation"]
    assert [rel.label for rel in documents[1].rels] == ["no_relation", "no_relation"]
    assert [rel.label for rel in documents[2].rels] == ["per:children", "per:parents"]
