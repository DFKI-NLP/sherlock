from .entity_fishing import EntityFishingPredictor
from .predictor import Predictor
from .spacy import SpacyPredictor
from .transformers import TransformersBinaryRcPredictor, TransformersTokenClfPredictor


__all__ = [
    "Predictor",
    "TransformersBinaryRcPredictor",
    "TransformersTokenClfPredictor",
    "SpacyPredictor",
    "EntityFishingPredictor",
]
