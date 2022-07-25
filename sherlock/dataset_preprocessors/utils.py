from typing import Union, List

import logging
import spacy
import torch
from spacy.tokens import Doc
from spacy.vocab import Vocab

from uuid import uuid4
from collections import Counter


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


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
        logging.warning(f"NER model predicted O tag for [{doc[start:end]}] in: {doc} ")
    return entity_type
