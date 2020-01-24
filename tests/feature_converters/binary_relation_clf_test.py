import os

from tests import FIXTURES_ROOT

from transformers import BertTokenizer
from sherlock.dataset_readers import TacredDatasetReader
from sherlock.feature_converters import BinaryRelationClfConverter


def test_convert_documents_to_features():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRelationClfConverter(tokenizer=tokenizer,
                                           labels=reader.get_labels(task="binary_re"))

    documents = reader.get_documents(split="train")

    input_features = converter.documents_to_features(documents)

    assert len(input_features) == 3

    features = input_features[0]

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial", "officer",
                       "[head_start]", "douglas", "flint", "[head_end]", "will", "become",
                       "[tail_start]", "chairman", "[tail_end]", ",", "succeeding", "stephen",
                       "green", "who", "is", "leaving", "to", "take", "a", "government",
                       "job", ".", "[SEP]"]

    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_entity_handling_mark_entity():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRelationClfConverter(tokenizer=tokenizer,
                                           labels=reader.get_labels(task="binary_re"),
                                           entity_handling="mark_entity")

    documents = reader.get_documents(split="train")
    input_features = converter.documents_to_features(documents)
    features = input_features[0]

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial", "officer",
                       "[head_start]", "douglas", "flint", "[head_end]", "will", "become",
                       "[tail_start]", "chairman", "[tail_end]", ",", "succeeding", "stephen",
                       "green", "who", "is", "leaving", "to", "take", "a", "government",
                       "job", ".", "[SEP]"]

    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_entity_handling_mark_entity_append_ner():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRelationClfConverter(tokenizer=tokenizer,
                                           labels=reader.get_labels(task="binary_re"),
                                           entity_handling="mark_entity_append_ner")

    documents = reader.get_documents(split="train")
    input_features = converter.documents_to_features(documents)
    features = input_features[0]

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial", "officer",
                       "[head_start]", "douglas", "flint", "[head_end]", "will", "become",
                       "[tail_start]", "chairman", "[tail_end]", ",", "succeeding", "stephen",
                       "green", "who", "is", "leaving", "to", "take", "a", "government",
                       "job", ".", "[SEP]", "[head=person]", "[SEP]", "[tail=title]", "[SEP]"]

    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_entity_handling_mask_entity():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRelationClfConverter(tokenizer=tokenizer,
                                           labels=reader.get_labels(task="binary_re"),
                                           entity_handling="mask_entity")

    documents = reader.get_documents(split="train")
    input_features = converter.documents_to_features(documents)
    features = input_features[0]

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial", "officer",
                       "[head=person]", "will", "become", "[tail=title]", ",", "succeeding",
                       "stephen", "green", "who", "is", "leaving", "to", "take", "a",
                       "government", "job", ".", "[SEP]"]

    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_entity_handling_mask_entity_append_text():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRelationClfConverter(tokenizer=tokenizer,
                                           labels=reader.get_labels(task="binary_re"),
                                           entity_handling="mask_entity_append_text")

    documents = reader.get_documents(split="train")
    input_features = converter.documents_to_features(documents)
    features = input_features[0]

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial", "officer",
                       "[head=person]", "will", "become", "[tail=title]", ",", "succeeding",
                       "stephen", "green", "who", "is", "leaving", "to", "take", "a",
                       "government", "job", ".", "[SEP]", "douglas", "flint", "[SEP]",
                       "chairman", "[SEP]"]

    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens
