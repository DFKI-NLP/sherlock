import os

from tests import FIXTURES_ROOT

from transformers import BertTokenizer
from sherlock.dataset_readers import TacredDatasetReader
from sherlock.feature_converters import NerConverter


def test_create_converter():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="ner"))
    converter = NerConverter(tokenizer=tokenizer,
                             labels=reader.get_labels(task="ner"))

    assert converter.pad_token_label_id == -100
    assert len(converter.label_to_id_map) == len(reader.get_labels(task="ner")) == 17
    assert len(converter.id_to_label_map) == len(reader.get_labels(task="ner"))


def test_convert_documents_to_features():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="ner"))
    converter = NerConverter(tokenizer=tokenizer,
                             labels=reader.get_labels(task="ner"),
                             log_num_input_features=1)

    documents = reader.get_documents(split="train")

    input_features = converter.documents_to_features(documents)
    assert len(input_features) == 3

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial", "officer",
                       "douglas", "flint", "will", "become", "chairman", ",", "succeeding",
                       "stephen", "green", "who", "is", "leaving", "to", "take", "a",
                       "government", "job", ".", "[SEP]"]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens

    label_ids = input_features[1].labels
    assert label_ids[0] == converter.pad_token_label_id
    assert label_ids[-1] == converter.pad_token_label_id
    assert label_ids[1:8] == ([converter.label_to_id_map["B-LOCATION"]]
                              + [converter.pad_token_label_id] * 3
                              + [0] * 3)
    
    assert len(features.input_ids) == converter.max_length
    assert len(features.input_ids) == converter.max_length
    assert len(features.attention_mask) == converter.max_length
    assert len(features.token_type_ids) == converter.max_length
    assert len(features.labels) == converter.max_length


def test_convert_documents_to_features_truncate():
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")

    max_length = 10
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens(task="ner"))
    converter = NerConverter(tokenizer=tokenizer,
                             labels=reader.get_labels(task="ner"),
                             max_length=max_length)

    documents = reader.get_documents(split="train")

    input_features = converter.documents_to_features(documents)
    assert all([features.metadata["truncated"] for features in input_features])

    expected_tokens = ["[CLS]", "at", "the", "same", "time", ",", "chief", "financial",
                       "officer", "[SEP]"]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens

    label_ids = input_features[1].labels
    assert label_ids[0] == converter.pad_token_label_id
    assert label_ids[-1] == converter.pad_token_label_id

    assert len(features.input_ids) == max_length
    assert len(features.input_ids) == max_length
    assert len(features.attention_mask) == max_length
    assert len(features.token_type_ids) == max_length
    assert len(features.labels) == max_length


def test_save_and_load(tmpdir):
    reader = TacredDatasetReader(data_dir=os.path.join(FIXTURES_ROOT, "datasets"),
                                 train_file="tacred.json")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    converter = NerConverter(tokenizer=tokenizer,
                             labels=reader.get_labels(task="ner"),
                             max_length=1,
                             pad_token_segment_id=2,
                             pad_token_label_id=3,
                             log_num_input_features=4)
    converter.save(tmpdir)

    loaded_converter = NerConverter.from_pretrained(tmpdir, tokenizer)
    assert loaded_converter.max_length == converter.max_length
    assert loaded_converter.pad_token_segment_id == converter.pad_token_segment_id
    assert loaded_converter.pad_token_label_id == converter.pad_token_label_id
    assert loaded_converter.label_to_id_map == converter.label_to_id_map
    assert loaded_converter.id_to_label_map == converter.id_to_label_map
    assert loaded_converter.log_num_input_features == -1
