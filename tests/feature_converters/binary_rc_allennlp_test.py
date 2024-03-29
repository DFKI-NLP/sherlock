import json
import os

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from sherlock.dataset_readers import TacredDatasetReader
from sherlock.feature_converters import BinaryRcConverter
from sherlock.tasks import IETask
from tests import FIXTURES_ROOT


TRAIN_FILE = os.path.join(FIXTURES_ROOT, "datasets", "tacred.json")
# TODO: create token_classification_allennlp test

def test_convert_documents_to_features():
    reader = TacredDatasetReader()

    tokenizer = PretrainedTransformerTokenizer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 512, "add_special_tokens": false,"tokenizer_kwargs": {"use_fast": false}}""")))
    tokenizer.tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    token_indexer = PretrainedTransformerIndexer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 512,"tokenizer_kwargs": {"use_fast": false}}""")))

    converter = BinaryRcConverter(
        max_length=512,
        framework="allennlp",
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
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

    tokens = [i.text for i in features.instance["text"].tokens]
    assert tokens == expected_tokens

    # use <= since in AllenNLP, Instance objects are not padded, this happens later in
    # https://docs.allennlp.org/main/api/data/batch/
    assert len(features.instance["text"]) <= converter.max_length


def test_convert_documents_to_features_truncate():
    ## Expected behavior: cut off all instances where an entity is cut off,
    ## but allow cut off of non-entity tokens.
    reader = TacredDatasetReader()

    max_length = 19
    tokenizer = PretrainedTransformerTokenizer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": """ + str(max_length)
        + """, "add_special_tokens": false, "tokenizer_kwargs": {"use_fast": false}}""")))
    tokenizer.tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    token_indexer = PretrainedTransformerIndexer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": """ + str(max_length)
        + """, "tokenizer_kwargs": {"use_fast": false}}""")))

    converter = BinaryRcConverter(
        max_length=max_length,
        framework="allennlp",
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        log_num_input_features=1,
    )

    # TODO: once generator support for FeatureConverts: remove list()
    documents = list(reader.get_documents(file_path=TRAIN_FILE))

    input_features = converter.documents_to_features(documents)

    # The second and third example are too long.
    # => only want one output example.
    assert len(input_features) == 1

    features = input_features[0]

    expected_tokens = [
        '[CLS]',
        'at',
        'the',
        'same',
        'time',
        ',',
        'chief',
        'financial',
        'officer',
        '[head_start]',
        'douglas',
        'flint',
        '[head_end]',
        'will',
        'become',
        '[tail_start]',
        'chairman',
        '[tail_end]',
        '[SEP]',
    ]

    tokens = [i.text for i in features.instance["text"].tokens]
    assert tokens == expected_tokens
    assert features.metadata["truncated"]
    assert features.metadata["guid"] == "e7798fb926b9403cfcd2"
    assert features.metadata["head_idx"] == 0
    assert features.metadata["tail_idx"] == 1

    # use <= since in AllenNLP, Instance objects are not padded, this happens later in
    # https://docs.allennlp.org/main/api/data/batch/
    assert len(features.instance["text"]) <= converter.max_length

    ## Now check if it truncates if tail is cut off.
    reader = TacredDatasetReader()

    max_length = 18
    tokenizer = PretrainedTransformerTokenizer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": """ + str(max_length)
        + """, "add_special_tokens": false, "tokenizer_kwargs": {"use_fast": false}}""")))
    tokenizer.tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    token_indexer = PretrainedTransformerIndexer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": """ + str(max_length)
        + """, "tokenizer_kwargs": {"use_fast": false}}""")))

    converter = BinaryRcConverter(
        max_length=max_length,
        framework="allennlp",
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        log_num_input_features=1,
    )

    # TODO: once generator support for FeatureConverts: remove list()
    documents = list(reader.get_documents(file_path=TRAIN_FILE))

    input_features = converter.documents_to_features(documents)

    assert len(input_features) == 0

    ## Now check for "mark_entity_append_ner" mode
    reader = TacredDatasetReader()

    max_length = 19
    tokenizer = PretrainedTransformerTokenizer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": """ + str(max_length)
        + """, "add_special_tokens": false, "tokenizer_kwargs": {"use_fast": false}}""")))
    tokenizer.tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    token_indexer = PretrainedTransformerIndexer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": """ + str(max_length)
        + """, "tokenizer_kwargs": {"use_fast": false}}""")))

    converter = BinaryRcConverter(
        max_length=max_length,
        framework="allennlp",
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        entity_handling="mark_entity_append_ner",
        log_num_input_features=1,
    )

    # TODO: once generator support for FeatureConverts: remove list()
    documents = list(reader.get_documents(file_path=TRAIN_FILE))

    input_features = converter.documents_to_features(documents)

    assert len(input_features) == 0


def test_entity_handling_mark_entity():
    reader = TacredDatasetReader()

    tokenizer = PretrainedTransformerTokenizer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 512, "add_special_tokens": false,"tokenizer_kwargs": {"use_fast": false}}""")))
    tokenizer.tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    token_indexer = PretrainedTransformerIndexer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 512, "tokenizer_kwargs": {"use_fast": false}}""")))

    converter = BinaryRcConverter(
        framework="allennlp",
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
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

    tokens = [i.text for i in features.instance["text"].tokens]
    assert tokens == expected_tokens


def test_entity_handling_mark_entity_append_ner():
    reader = TacredDatasetReader()

    tokenizer = PretrainedTransformerTokenizer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 512, "add_special_tokens": false,"tokenizer_kwargs": {"use_fast": false}}""")))
    tokenizer.tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    token_indexer = PretrainedTransformerIndexer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 512, "tokenizer_kwargs": {"use_fast": false}}""")))

    converter = BinaryRcConverter(
        max_length=512,
        framework="allennlp",
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        log_num_input_features=1,
        entity_handling="mark_entity_append_ner",
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
        "[head=person]",
        "[SEP]",
        "[tail=title]",
        "[SEP]",
    ]

    tokens = [i.text for i in features.instance["text"].tokens]
    assert tokens == expected_tokens


def test_entity_handling_mask_entity():
    reader = TacredDatasetReader()

    tokenizer = PretrainedTransformerTokenizer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 512, "add_special_tokens": false,"tokenizer_kwargs": {"use_fast": false}}""")))
    tokenizer.tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    token_indexer = PretrainedTransformerIndexer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 512, "tokenizer_kwargs": {"use_fast": false}}""")))

    converter = BinaryRcConverter(
        max_length=512,
        framework="allennlp",
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        log_num_input_features=1,
        entity_handling="mask_entity",
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

    tokens = [i.text for i in features.instance["text"].tokens]
    assert tokens == expected_tokens


def test_entity_handling_mask_entity_append_text():
    reader = TacredDatasetReader()

    tokenizer = PretrainedTransformerTokenizer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 512, "add_special_tokens": false,"tokenizer_kwargs": {"use_fast": false}}""")))
    tokenizer.tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    token_indexer = PretrainedTransformerIndexer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 512, "tokenizer_kwargs": {"use_fast": false}}""")))

    converter = BinaryRcConverter(
        max_length=512,
        framework="allennlp",
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        log_num_input_features=1,
        entity_handling="mask_entity_append_text",
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

    tokens = [i.text for i in features.instance["text"].tokens]
    assert tokens == expected_tokens


def test_save_and_load(tmpdir):
    reader = TacredDatasetReader()

    tokenizer = PretrainedTransformerTokenizer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 10, "add_special_tokens": false, "tokenizer_kwargs": {"use_fast": false}}""")))
    tokenizer.tokenizer.add_tokens(
        reader.get_additional_tokens(IETask.BINARY_RC, file_path=TRAIN_FILE))
    token_indexer = PretrainedTransformerIndexer.from_params(Params(json.loads(
        """{"model_name": "bert-base-uncased", "max_length": 10, "tokenizer_kwargs": {"use_fast": false}}""")))

    converter = BinaryRcConverter(
        max_length=10,
        framework="allennlp",
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        log_num_input_features=3,
        entity_handling="mask_entity_append_text",
    )

    converter.save(tmpdir)
    loaded_converter = BinaryRcConverter.from_pretrained(
        tmpdir, tokenizer=tokenizer, token_indexers={"tokens": token_indexer})
    assert loaded_converter.max_length == converter.max_length
    assert loaded_converter.entity_handling == converter.entity_handling
    assert loaded_converter.sep_token == converter.sep_token
    assert loaded_converter.lower_cases == converter.lower_cases
    assert loaded_converter.n_special_tokens == converter.n_special_tokens
    assert loaded_converter.log_num_input_features == -1
