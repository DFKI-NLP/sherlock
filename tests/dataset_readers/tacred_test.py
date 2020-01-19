from tests import FIXTURES_ROOT

from sherlock.dataset_readers import TacredDatasetReader


def test_get_documents():
    reader = TacredDatasetReader(data_dir=FIXTURES_ROOT, train_file="tacred.json")

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


def test_get_labels():
    reader = TacredDatasetReader(data_dir=FIXTURES_ROOT, train_file="tacred.json")

    labels = reader.get_labels()
    assert labels == ["no_relation", "per:title", "per:city_of_death"]


def test_additional_tokens():
    reader = TacredDatasetReader(data_dir=FIXTURES_ROOT, train_file="tacred.json")

    additional_tokens = reader.get_additional_tokens()
    print(additional_tokens)

    assert sorted(additional_tokens) == sorted(["[HEAD_START]", "[HEAD_END]", "[TAIL_START]", "[TAIL_END]",
                                                "[HEAD=PERSON]", "[TAIL=TITLE]", "[TAIL=PERSON]", "[TAIL=CITY]"])
