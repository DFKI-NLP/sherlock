import os
import logging
import gzip
import spacy
import torch

from typing import Union, List
from uuid import uuid4
from collections import Counter
from spacy.tokens import Doc
from spacy.vocab import Vocab
from allennlp.data.dataset_readers.dataset_utils import span_utils


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def open_file(filename, mode, encoding="utf-8"):
    if ".gz" in os.path.splitext(filename)[1]:
        if mode == "w" or mode == "a":
            mode += "t"     # text mode
        return gzip.open(filename, mode)
    else:
        return open(filename, mode=mode, encoding=encoding)

def generate_example_id():
    return str(uuid4())


def swap_args(example):
    example["entities"][0], example["entities"][1] = example["entities"][1], example["entities"][0]
    if "type" in example:
        example["type"][0], example["type"][1] = example["type"][1], example["type"][0]
    return example


def get_label_counter(examples):
    labels = [example["label"] for example in examples]
    return dict(Counter(labels))


class _PretokenizedTokenizer:
    """
    Custom tokenizer to be used in spaCy when the text is already pretokenized.
    https://github.com/explosion/spaCy/issues/5399#issuecomment-624171591
    """

    def __init__(self, vocab: Vocab):
        """Initialize tokenizer with a given vocab
        :param vocab: an existing vocabulary (see https://spacy.io/api/vocab)
        """
        self.vocab = vocab

    def __call__(self, inp: Union[List[str], str]) -> Doc:
        """Call the tokenizer on input `inp`.
        :param inp: either a string to be split on whitespace, or a list of tokens
        :return: the created Doc object
        """
        if isinstance(inp, str):
            words = inp.split()
            spaces = [True] * (len(words) - 1) + (
                [True] if inp[-1].isspace() else [False]
            )
            return Doc(self.vocab, words=words, spaces=spaces)
        elif isinstance(inp, list):
            return Doc(self.vocab, words=inp)
        else:
            raise ValueError(
                "Unexpected input format. Expected string to be split on whitespace, or list of tokens."
            )


def load_spacy_predictor(saved_model_path, cuda_device: int = 0):
    if cuda_device > -1 and torch.cuda.is_available():
        spacy.prefer_gpu(cuda_device)
    # load the trained model
    model = spacy.load(saved_model_path)
    # set custom tokenizer to preserve existing tokenization
    model.tokenizer = _PretokenizedTokenizer(model.vocab)
    return model


def get_entity_type(doc, start, end):
    entity_tokens = [t for t in doc][start:end]
    entity_type = "O"
    for t in entity_tokens:
        # use entity tag of the first token with no O tag, which is hopefully the head token
        if t.ent_iob_ != "O":
            entity_type = t.ent_type_
            break
    if entity_type == "O":
        logging.debug(f"NER model predicted O tag for [{doc[start:end]}] in: {doc} ")
    return entity_type


def predict_entity_type(spacy_ner_predictor, examples, batch_size=1000):
    if spacy_ner_predictor is not None:
        tokens_list = [example["tokens"] for example in examples]
        i = 0
        for doc in spacy_ner_predictor.pipe(tokens_list, batch_size=batch_size):
            subj_start, subj_end = examples[i]["entities"][0]
            obj_start, obj_end = examples[i]["entities"][1]
            subj_type = get_entity_type(doc, subj_start, subj_end)
            obj_type = get_entity_type(doc, obj_start, obj_end)
            examples[i]["type"] = [subj_type, obj_type]
            i += 1
    return examples


def get_entities(ner_labels: List[str], tagging_format: str = "bio") -> List[dict]:
    """
    Given a sequence corresponding to e.g. BIO tags, extracts named entities.

    Parameters
    ----------
    ner_labels : List[str]
        Sequence of NER tags
    tagging_format : str, default="bio"
        Used to determine which span util function to use

    Returns
    ----------
    entities : List[dict]
        List of entity dictionaries with spans and entity label
    """
    assert tagging_format in [
        "bio",
        "iob1",
        "bioul",
    ], "Valid tagging format options are ['bio', 'iob1', 'bioul']"
    if tagging_format == "iob1":
        tags_to_spans = span_utils.iob1_tags_to_spans
    elif tagging_format == "bioul":
        tags_to_spans = span_utils.bioul_tags_to_spans
    else:
        tags_to_spans = span_utils.bio_tags_to_spans

    typed_string_spans = tags_to_spans(ner_labels)
    entities = []
    for label, span in typed_string_spans:
        entities.append(
            {
                "start": span[0],
                "end": span[1] + 1,  # make span exclusive
                "label": label,
            }
        )
    entities.sort(key=lambda e: e["start"])
    return entities
