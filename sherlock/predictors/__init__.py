from .entity_fishing import EntityFishingPredictor
from .spacy import SpacyPredictor
from .transformers import TransformersBinaryRcPredictor, TransformersTokenClfPredictor


__all__ = [
    "TransformersBinaryRcPredictor",
    "TransformersTokenClfPredictor",
    "SpacyPredictor",
    "EntityFishingPredictor",
]
