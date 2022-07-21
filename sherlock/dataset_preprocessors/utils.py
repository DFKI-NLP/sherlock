from typing import Union, List

import spacy
import torch
from spacy.tokens import Doc
from spacy.vocab import Vocab

from uuid import uuid4
from collections import Counter


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
    head_token = entity_tokens[0]   # use entity tag of the head token
    assert head_token.ent_iob_ != "O", "NER model predicted O tag for the head token"
    return head_token.ent_type_
