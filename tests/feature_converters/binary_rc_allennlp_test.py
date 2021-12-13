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


def test_convert_documents_to_features():
    reader = TacredDatasetReader(
        data_dir=os.path.join(FIXTURES_ROOT, "datasets"), train_file="tacred.json"
    )

    tokenizer = PretrainedTransformerTokenizer.from_params(Params(json.loads("""{"model_name": "bert-base-uncased"}""")))
    tokenizer.tokenizer.add_tokens(reader.get_additional_tokens(IETask.BINARY_RC))
    token_indexer = PretrainedTransformerIndexer.from_params(Params(json.loads("""{"model_name": "bert-base-uncased"}""")))
    vocab = Vocabulary()
    # manually add the labels seen in fixtures/datasets/tacred.json to label Vocab
    for label in reader.get_labels(IETask.BINARY_RC):
        vocab.add_token_to_namespace(label, "labels")

    converter = BinaryRcConverter(labels=reader.get_labels(IETask.BINARY_RC), max_length=512,
                                  framework="allennlp", tokenizer=tokenizer,
                                  token_indexer=token_indexer, vocabulary=vocab, log_num_input_features=1)

    documents = reader.get_documents(split="train")

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
    # head_start is split currently, shouldnt, I guess
    tokens = [i for i in features.instance["text"].tokens]
    assert tokens == expected_tokens

    assert len(features.input_ids) == converter.max_length
    assert len(features.input_ids) == converter.max_length
    assert len(features.attention_mask) == converter.max_length
    assert len(features.token_type_ids) == converter.max_length

test_convert_documents_to_features()