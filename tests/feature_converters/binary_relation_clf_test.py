from tests import FIXTURES_ROOT

from transformers import BertTokenizer
from sherlock.dataset_readers import TacredDatasetReader
from sherlock.feature_converters import BinaryRelationClfConverter


def test_convert_documents_to_features():
    reader = TacredDatasetReader(data_dir=FIXTURES_ROOT, train_file="tacred.json")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(reader.get_additional_tokens())
    converter = BinaryRelationClfConverter(tokenizer=tokenizer,
                                           labels=reader.get_labels())

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
