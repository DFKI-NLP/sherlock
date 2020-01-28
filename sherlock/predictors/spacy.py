import logging
from typing import Dict, List, Tuple

import spacy
from spacy.cli.download import download as spacy_download
from spacy.language import Language as SpacyModelType

from sherlock import Document
from sherlock.document import Token
from sherlock.predictors import Predictor


logger = logging.getLogger(__name__)


LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool], SpacyModelType] = {}


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
    return [token for token in tokens if not token.is_space]


@Predictor.register("spacy")
class SpacyPredictor(Predictor):
    def __init__(
        self,
        language: str = "en_core_web_sm",
        pos_tags: bool = False,
        parse: bool = False,
        ner: bool = False,
        split_on_spaces: bool = False,
    ) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)

    @classmethod
    def from_pretrained(  # type: ignore
        cls,
        language: str = "en_core_web_sm",
        pos_tags: bool = False,
        parse: bool = False,
        ner: bool = False,
        split_on_spaces: bool = False,
    ) -> "Predictor":  # type: ignore
        return cls(language, pos_tags, parse, ner, split_on_spaces)

    def predict_documents(self, documents: List[Document]) -> List[Document]:
        doc_tokens = [
            _remove_spaces(tokens)
            for tokens in self.spacy.pipe([doc.text for doc in documents], n_threads=-1)
        ]

        for document, tokens in zip(documents, doc_tokens):
            document.tokens = [Token.from_spacy(document, token) for token in tokens]
        return documents

    def predict_document(self, document: Document) -> Document:
        document.tokens = [
            Token.from_spacy(document, token) for token in _remove_spaces(self.spacy(document.text))
        ]
        return document
