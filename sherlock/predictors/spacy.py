import logging
import re
from typing import Dict, List, Tuple

import spacy
from spacy.cli.download import download as spacy_download
from spacy.language import Language as SpacyModelType

from sherlock import Document
from sherlock.document import Span, Token
from sherlock.predictors.predictor import Predictor


logger = logging.getLogger(__name__)


LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool], SpacyModelType] = {}
SPACY_ESCAPE_CHAR_REGEX = re.compile(r"[\t\n\r\f\v]") # we replace these escape chars with "_" before running spacy,
# so that they get properly tokenized, and discard the corresponding tokens afterwards. Otherwise, we will get
# harder-to-remove spacy tokens such as '\n \n' or '\t \n \n'. Note: Currently not used.
WHITESPACE_ONLY_REGEX = re.compile(r"^[ \t\n\r\f\v]+$")

# taken from: https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py
def get_spacy_model(
    spacy_model_name: str, pos_tags: bool, parse: bool, ner: bool
) -> SpacyModelType:
    """
    In order to avoid loading spacy models a whole bunch of times, we'll save references to them,
    keyed by the options we used to create the spacy model, so any particular configuration only
    gets loaded once.
    """

    options = (spacy_model_name, pos_tags, parse, ner)
    if options not in LOADED_SPACY_MODELS:
        disable = ["vectors", "textcat"]
        if not pos_tags:
            disable.append("tagger")
        if not parse:
            disable.append("parser")
        if not ner:
            disable.append("ner")
        try:
            spacy_model = spacy.load(spacy_model_name, disable=disable)
        except OSError:
            logger.warning(
                f"Spacy models '{spacy_model_name}' not found.  Downloading and installing."
            )
            spacy_download(spacy_model_name)

            # Import the downloaded model module directly and load from there
            spacy_model_module = __import__(spacy_model_name)
            spacy_model = spacy_model_module.load(disable=disable)  # type: ignore

        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]


def _remove_spaces(tokens_isspace: List[Tuple[Token, bool]]) -> List[Token]:
    """
    Copied from AllenNLP's SpacyTokenizer at
    https://github.com/allenai/allennlp/blob/main/allennlp/data/tokenizers/spacy_tokenizer.py,
    but also updates the dep head indices to account for removed tokens.
    """
    tokens = []
    is_space_count = 0
    for (token, is_space) in tokens_isspace:
        if is_space:
            is_space_count += 1
        else:
            token.dep_head -= is_space_count
            tokens.append(token)
    return tokens


def _remove_escape_char_and_whitespace_tokens(tokens: List[Token]) -> List[Token]:
    """ Currently not used """
    def is_whitespace(token: Token) -> bool:
        return token.lemma == ' ' and token.tag == "_SP"
    def is_escape_char(token: Token) -> bool:
        return token.lemma == '_' and SPACY_ESCAPE_CHAR_REGEX.match(token.doc.text[token.start:token.end])
    return [token for token in tokens if not (is_whitespace(token) or is_escape_char(token))]


def _replace_ws(text: str) -> str:
    """
    Replace escape characters for easier tokenization by Spacy. Currently not used.
    """
    return SPACY_ESCAPE_CHAR_REGEX.sub("_", text)


def _is_empty_sentence(sent:spacy.tokens.Span) -> bool:
    """

    Parameters
    ----------
    sent  - a sentence

    Returns
    -------
    True if the sentence's text consists of whitespace-like characters only.
    """
    return WHITESPACE_ONLY_REGEX.match(sent.text) is not None


def _convert_sents(spacy_doc: spacy.tokens.Doc, tokens_isspace: List[Tuple[spacy.tokens.Token, bool]], doc:Document) -> List[Span]:
    sents = []
    for sent in spacy_doc.sents:
        if not _is_empty_sentence(sent):
            # fix start, end to account for removed space tokens before and at start/end of sentence
            sent_start = sent.start
            sent_end = sent.end
            while spacy_doc[sent_start].is_space and sent_start < sent_end:
                sent_start += 1
            while spacy_doc[sent_end - 1].is_space and sent_end > sent_start:
                sent_end -= 1
            sent_start -= len([is_space for (token, is_space) in tokens_isspace[:sent_start] if is_space])
            sent_end -= len([is_space for (token, is_space) in tokens_isspace[:sent_end] if is_space])
            sents.append(Span(doc, sent_start, sent_end))
    return sents


@Predictor.register("spacy")
class SpacyPredictor(Predictor):
    def __init__(
        self,
        path: str = "en_core_web_sm",
        pos_tags: bool = False,
        parse: bool = False,
        ner: bool = False,
        split_on_spaces: bool = False,
    ) -> None:
        self.spacy = get_spacy_model(path, pos_tags, parse, ner)
        self.has_sentencizer = parse

    @classmethod
    def from_pretrained(  # type: ignore
        cls,
        path: str,
        **kwargs,
        # path: str = "en_core_web_sm",
        # pos_tags: bool = False,
        # parse: bool = False,
        # ner: bool = False,
        # split_on_spaces: bool = False,
    ) -> "Predictor":  # type: ignore
        return cls(
            path,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ["pos_tags", "parse", "ner", "split_on_spaces"]
            },
        )

    def predict_documents(self, documents: List[Document]) -> List[Document]:
        # following AllenAI's usage of Spacy Tokenizer, i.e. removing spaces after tokenization
        spacy_docs = self.spacy.pipe([doc.text for doc in documents], n_threads=-1)
        for doc, spacy_doc in zip(documents, spacy_docs):
            tokens_isspace = [(Token.from_spacy(doc, token), token.is_space) for token in spacy_doc]
            doc.tokens = _remove_spaces(tokens_isspace)
            if self.has_sentencizer:
                doc.sents = _convert_sents(spacy_doc, tokens_isspace, doc)
            for mention in spacy_doc.ents:
                # todo account for removed space tokens?
                doc.ments.append(Span.from_spacy(doc, mention))
        return documents

    def predict_document(self, document: Document) -> Document:
        # following AllenAI's usage of Spacy Tokenizer, i.e. removing spaces after tokenization
        spacy_doc = self.spacy(document.text)
        tokens_isspace = [(Token.from_spacy(document, token), token.is_space) for token in spacy_doc]
        document.tokens = _remove_spaces(tokens_isspace)
        if self.has_sentencizer:
            document.sents = _convert_sents(spacy_doc, tokens_isspace, document)
        for mention in spacy_doc.ents:
            # todo account for removed space tokens?
            document.ments.append(Span.from_spacy(document, mention))
        return document
