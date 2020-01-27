import os

from sherlock.dataset_readers import Conll2003DatasetReader
from sherlock.tasks import IETask
from tests import FIXTURES_ROOT


def test_get_documents():
    reader = Conll2003DatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"), train_file="conll.txt"
    )

    documents = reader.get_documents(split="train")
    assert len(documents) == 8

    doc = documents[0]
    assert len(doc.tokens) == 9
    assert len(doc.ments) == 3
    assert len(doc.rels) == 0

    assert [t.text for t in doc.tokens] == [
        "EU",
        "rejects",
        "German",
        "call",
        "to",
        "boycott",
        "British",
        "lamb",
        ".",
    ]

    expected_mentions = [
        dict(start=0, end=1, label="ORG"),
        dict(start=2, end=3, label="MISC"),
        dict(start=6, end=7, label="MISC"),
    ]

    for ment, expected_ment in zip(sorted(doc.ments, key=lambda m: m.start), expected_mentions):
        assert ment.start == expected_ment["start"]
        assert ment.end == expected_ment["end"]
        assert ment.label == expected_ment["label"]


def test_get_labels():
    reader = Conll2003DatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"), train_file="conll.txt"
    )

    labels = reader.get_labels(IETask.NER)
    assert list(sorted(labels)) == list(sorted(["O", "I-MISC", "I-PER", "I-ORG", "I-LOC"]))


def test_get_additional_tokens():
    reader = Conll2003DatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"), train_file="conll.txt"
    )

    assert len(reader.get_additional_tokens(IETask.NER)) == 0
