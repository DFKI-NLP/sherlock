import os

from transformers import BertTokenizer

from sherlock.dataset_readers import TacredDatasetReader
from sherlock.feature_converters import BinaryRcConverter
from sherlock.tasks import IETask
from tests import FIXTURES_ROOT


TRAIN_FILE = os.path.join(FIXTURES_ROOT, "datasets", "tacred.json")


def test_convert_documents_to_features():

    reader = TacredDatasetReader()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    converter = BinaryRcConverter(
        max_length=512,
        tokenizer=tokenizer,
        labels=reader.get_labels(IETask.BINARY_RC, file_path=TRAIN_FILE),
        log_num_input_features=1,
    )

    # TODO: once generator support for FeatureConverts: remove list()
    documents = list(reader.get_documents(file_path=TRAIN_FILE))

    input_features = converter.documents_to_features(documents)

    assert len(input_features) == 3

    features = input_features[0]

    expected_tokens = [
        "[CLS]",
        "at",
        "the",
        "same",
        "time",
        ",",
        "chief",
        "financial",
        "officer",
        "[head_start]",
        "douglas",
        "flint",
        "[head_end]",
        "will",
        "become",
        "[tail_start]",
        "chairman",
        "[tail_end]",
        ",",
        "succeeding",
        "stephen",
        "green",
        "who",
        "is",
        "leaving",
        "to",
        "take",
        "a",
        "government",
        "job",
        ".",
        "[SEP]",
    ]

    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens

    assert len(features.input_ids) == converter.max_length
    assert len(features.input_ids) == converter.max_length
    assert len(features.attention_mask) == converter.max_length
    assert len(features.token_type_ids) == converter.max_length


def test_convert_documents_to_features_truncate():
    reader = TacredDatasetReader()

    max_length = 19
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    converter = BinaryRcConverter(
        tokenizer=tokenizer,
        labels=reader.get_labels(IETask.BINARY_RC, file_path=TRAIN_FILE),
        max_length=max_length,
    )

    # TODO: once generator support for FeatureConverts: remove list()
    documents = list(reader.get_documents(file_path=TRAIN_FILE))

    input_features = converter.documents_to_features(documents)
    assert all([features.metadata["truncated"] for features in input_features])

    expected_tokens = [
        "[CLS]",
        "at",
        "the",
        "same",
        "time",
        ",",
        "chief",
        "financial",
        "officer",
        '[head_start]',
        "douglas",
        "flint",
        "[head_end]",
        "will",
        "become",
        "[tail_start]",
        "chairman",
        "[tail_end]",
        "[SEP]",
    ]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens

    assert len(features.input_ids) == max_length
    assert len(features.input_ids) == max_length
    assert len(features.attention_mask) == max_length
    assert len(features.token_type_ids) == max_length


    ## Check truncation boundary
    max_length = 18
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    converter = BinaryRcConverter(
        tokenizer=tokenizer,
        labels=reader.get_labels(IETask.BINARY_RC, file_path=TRAIN_FILE),
        max_length=max_length,
    )

    # TODO: once generator support for FeatureConverts: remove list()
    documents = list(reader.get_documents(file_path=TRAIN_FILE))

    input_features = converter.documents_to_features(documents)

    assert len(input_features) == 0


def test_entity_handling_mark_entity():
    reader = TacredDatasetReader()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    converter = BinaryRcConverter(
        tokenizer=tokenizer,
        labels=reader.get_labels(IETask.BINARY_RC, file_path=TRAIN_FILE),
        entity_handling="mark_entity",
    )

    # TODO: once generator support for FeatureConverts: remove list()
    documents = list(reader.get_documents(file_path=TRAIN_FILE))
    input_features = converter.documents_to_features(documents)

    expected_tokens = [
        "[CLS]",
        "at",
        "the",
        "same",
        "time",
        ",",
        "chief",
        "financial",
        "officer",
        "[head_start]",
        "douglas",
        "flint",
        "[head_end]",
        "will",
        "become",
        "[tail_start]",
        "chairman",
        "[tail_end]",
        ",",
        "succeeding",
        "stephen",
        "green",
        "who",
        "is",
        "leaving",
        "to",
        "take",
        "a",
        "government",
        "job",
        ".",
        "[SEP]",
    ]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_entity_handling_mark_entity_append_ner():
    reader = TacredDatasetReader()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    converter = BinaryRcConverter(
        tokenizer=tokenizer,
        labels=reader.get_labels(IETask.BINARY_RC, file_path=TRAIN_FILE),
        entity_handling="mark_entity_append_ner",
    )

    # TODO: once generator support for FeatureConverts: remove list()
    documents = list(reader.get_documents(file_path=TRAIN_FILE))
    input_features = converter.documents_to_features(documents)

    expected_tokens = [
        "[CLS]",
        "at",
        "the",
        "same",
        "time",
        ",",
        "chief",
        "financial",
        "officer",
        "[head_start]",
        "douglas",
        "flint",
        "[head_end]",
        "will",
        "become",
        "[tail_start]",
        "chairman",
        "[tail_end]",
        ",",
        "succeeding",
        "stephen",
        "green",
        "who",
        "is",
        "leaving",
        "to",
        "take",
        "a",
        "government",
        "job",
        ".",
        "[SEP]",
        "[head=person]",
        "[SEP]",
        "[tail=title]",
        "[SEP]",
    ]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_entity_handling_mask_entity():
    reader = TacredDatasetReader()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    converter = BinaryRcConverter(
        tokenizer=tokenizer,
        labels=reader.get_labels(IETask.BINARY_RC, file_path=TRAIN_FILE),
        entity_handling="mask_entity",
    )

    # TODO: once generator support for FeatureConverts: remove list()
    documents = list(reader.get_documents(file_path=TRAIN_FILE))
    input_features = converter.documents_to_features(documents)

    expected_tokens = [
        "[CLS]",
        "at",
        "the",
        "same",
        "time",
        ",",
        "chief",
        "financial",
        "officer",
        "[head=person]",
        "will",
        "become",
        "[tail=title]",
        ",",
        "succeeding",
        "stephen",
        "green",
        "who",
        "is",
        "leaving",
        "to",
        "take",
        "a",
        "government",
        "job",
        ".",
        "[SEP]",
    ]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_entity_handling_mask_entity_append_text():
    reader = TacredDatasetReader()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    converter = BinaryRcConverter(
        tokenizer=tokenizer,
        labels=reader.get_labels(IETask.BINARY_RC, file_path=TRAIN_FILE),
        entity_handling="mask_entity_append_text",
    )

    # TODO: once generator support for FeatureConverts: remove list()
    documents = list(reader.get_documents(file_path=TRAIN_FILE))
    input_features = converter.documents_to_features(documents)

    expected_tokens = [
        "[CLS]",
        "at",
        "the",
        "same",
        "time",
        ",",
        "chief",
        "financial",
        "officer",
        "[head=person]",
        "will",
        "become",
        "[tail=title]",
        ",",
        "succeeding",
        "stephen",
        "green",
        "who",
        "is",
        "leaving",
        "to",
        "take",
        "a",
        "government",
        "job",
        ".",
        "[SEP]",
        "douglas",
        "flint",
        "[SEP]",
        "chairman",
        "[SEP]",
    ]

    features = input_features[0]
    tokens = tokenizer.convert_ids_to_tokens([i for i in features.input_ids if i != 0])
    assert tokens == expected_tokens


def test_save_and_load(tmpdir):
    reader = TacredDatasetReader()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    converter = BinaryRcConverter(
        tokenizer=tokenizer,
        labels=reader.get_labels(IETask.BINARY_RC, file_path=TRAIN_FILE),
        max_length=1,
        log_num_input_features=3,
        entity_handling="mask_entity_append_text",
    )
    converter.save(tmpdir)

    loaded_converter = BinaryRcConverter.from_pretrained(tmpdir, tokenizer=tokenizer)
    assert loaded_converter.max_length == converter.max_length
    assert loaded_converter.entity_handling == converter.entity_handling
    assert loaded_converter.label_to_id_map == converter.label_to_id_map
    assert loaded_converter.id_to_label_map == converter.id_to_label_map
    assert loaded_converter.log_num_input_features == -1
