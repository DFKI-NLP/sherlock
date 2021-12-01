from .entity_fishing import EntityFishingAnnotator
from .spacy import SpacyAnnotator
from .transformers import TransformersBinaryRcAnnotator, TransformersTokenClfAnnotator


__all__ = [
    "TransformersBinaryRcAnnotator",
    "TransformersTokenClfAnnotator",
    "SpacyAnnotator",
    "EntityFishingAnnotator",
]
