import os

from tests import FIXTURES_ROOT

from transformers import BertTokenizer
from sherlock.dataset_readers import TacredDatasetReader
from sherlock.feature_converters import BinaryRcConverter


def test_convert_documents_to_features():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRcConverter(tokenizer=tokenizer,
                                  labels=reader.get_labels(task="binary_re"),
                                  log_num_input_features=1)

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

    assert len(features.input_ids) == converter.max_length
    assert len(features.input_ids) == converter.max_length
    assert len(features.attention_mask) == converter.max_length
    assert len(features.token_type_ids) == converter.max_length


def test_convert_documents_to_features_truncate():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    max_length = 10
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRcConverter(tokenizer=tokenizer,
                                  labels=reader.get_labels(task="binary_re"),
                                  max_length=max_length)

    documents = reader.get_documents(split="train")

    input_features = converter.documents_to_features(documents)
    assert all([features.metadata["truncated"] for features in input_features])

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial",
                       "officer", "[SEP]"]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens

    assert len(features.input_ids) == max_length
    assert len(features.input_ids) == max_length
    assert len(features.attention_mask) == max_length
    assert len(features.token_type_ids) == max_length


def test_entity_handling_mark_entity():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRcConverter(tokenizer=tokenizer,
                                  labels=reader.get_labels(task="binary_re"),
                                  entity_handling="mark_entity")

    documents = reader.get_documents(split="train")
    input_features = converter.documents_to_features(documents)

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial", "officer",
                       "[head_start]", "douglas", "flint", "[head_end]", "will", "become",
                       "[tail_start]", "chairman", "[tail_end]", ",", "succeeding", "stephen",
                       "green", "who", "is", "leaving", "to", "take", "a", "government",
                       "job", ".", "[SEP]"]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_entity_handling_mark_entity_append_ner():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRcConverter(tokenizer=tokenizer,
                                  labels=reader.get_labels(task="binary_re"),
                                  entity_handling="mark_entity_append_ner")

    documents = reader.get_documents(split="train")
    input_features = converter.documents_to_features(documents)

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial", "officer",
                       "[head_start]", "douglas", "flint", "[head_end]", "will", "become",
                       "[tail_start]", "chairman", "[tail_end]", ",", "succeeding", "stephen",
                       "green", "who", "is", "leaving", "to", "take", "a", "government",
                       "job", ".", "[SEP]", "[head=person]", "[SEP]", "[tail=title]", "[SEP]"]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_entity_handling_mask_entity():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRcConverter(tokenizer=tokenizer,
                                  labels=reader.get_labels(task="binary_re"),
                                  entity_handling="mask_entity")

    documents = reader.get_documents(split="train")
    input_features = converter.documents_to_features(documents)

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial", "officer",
                       "[head=person]", "will", "become", "[tail=title]", ",", "succeeding",
                       "stephen", "green", "who", "is", "leaving", "to", "take", "a",
                       "government", "job", ".", "[SEP]"]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_entity_handling_mask_entity_append_text():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="binary_re"))
    converter = BinaryRcConverter(tokenizer=tokenizer,
                                  labels=reader.get_labels(task="binary_re"),
                                  entity_handling="mask_entity_append_text")

    documents = reader.get_documents(split="train")
    input_features = converter.documents_to_features(documents)

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial", "officer",
                       "[head=person]", "will", "become", "[tail=title]", ",", "succeeding",
                       "stephen", "green", "who", "is", "leaving", "to", "take", "a",
                       "government", "job", ".", "[SEP]", "douglas", "flint", "[SEP]",
                       "chairman", "[SEP]"]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_save_and_load(tmpdir):
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    converter = BinaryRcConverter(tokenizer=tokenizer,
                                  labels=reader.get_labels(task="binary_re"),
                                  max_length=1,
                                  pad_token_segment_id=2,
                                  log_num_input_features=3,
                                  entity_handling="mask_entity_append_text")
    converter.save(tmpdir)

    loaded_converter = BinaryRcConverter.from_pretrained(tmpdir, tokenizer)
    assert loaded_converter.max_length == converter.max_length
    assert loaded_converter.pad_token_segment_id == converter.pad_token_segment_id
    assert loaded_converter.entity_handling == converter.entity_handling
    assert loaded_converter.label_to_id_map == converter.label_to_id_map
    assert loaded_converter.id_to_label_map == converter.id_to_label_map
    assert loaded_converter.log_num_input_features == -1
