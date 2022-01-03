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
# harder-to-remove spacy tokens such as '\n \n' or '\t \n \n'
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


def _remove_spaces(tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
    """
    Copied from AllenNLP's SpacyTokenizer at
    https://github.com/allenai/allennlp/blob/main/allennlp/data/tokenizers/spacy_tokenizer.py
    """
    return [token for token in tokens if not token.is_space]


def _remove_escape_char_and_whitespace_tokens(tokens: List[Token]) -> List[Token]:
    def is_whitespace(token: Token) -> bool:
        return token.lemma == ' ' and token.tag == "_SP"
    def is_escape_char(token: Token) -> bool:
        return token.lemma == '_' and SPACY_ESCAPE_CHAR_REGEX.match(token.doc.text[token.start:token.end])
    return [token for token in tokens if not (is_whitespace(token) or is_escape_char(token))]


def _replace_ws(text: str) -> str:
    """
    Replace escape characters for easier tokenization by Spacy
    """
    return SPACY_ESCAPE_CHAR_REGEX.sub("_", text)

def _is_empty_sentence(sent:spacy.tokens.Span) -> bool:
    return WHITESPACE_ONLY_REGEX.match(sent.text) is not None

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
        #spacy_docs = self.spacy.pipe([_replace_ws(doc.text) for doc in documents], n_threads=-1)

        # following AllenAI's implementation of Spacy Tokenizer, i.e. removing spaces after tokenization
        spacy_docs = self.spacy.pipe([doc.text for doc in documents], n_threads=-1)
        for doc, spacy_doc in zip(documents, spacy_docs):
            doc.tokens = [Token.from_spacy(doc, token) for token in _remove_spaces([t for t in spacy_doc])]
            if self.has_sentencizer:
                # remove empty sentences
                doc.sents = [Span(doc, sent.start, sent.end) for sent in spacy_doc.sents if not _is_empty_sentence(sent)]
            for mention in spacy_doc.ents:
                doc.ments.append(Span.from_spacy(doc, mention))
        return documents

    def predict_document(self, document: Document) -> Document:
        spacy_doc = self.spacy(document.text)
        document.tokens = [Token.from_spacy(document, token) for token in _remove_spaces([t for t in spacy_doc])]
        if self.has_sentencizer:
            document.sents = [Span(document, sent.start, sent.end) for sent in spacy_doc.sents if not _is_empty_sentence(sent)]
        for mention in spacy_doc.ents:
            document.ments.append(Span.from_spacy(document, mention))
        return document
