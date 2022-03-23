from .entity_fishing import EntityFishingAnnotator
from .spacy import SpacyAnnotator
from .transformers import TransformersBinaryRcAnnotator, TransformersTokenClfAnnotator
from .allennlp import AllenNLPBinaryRcAnnotator

__all__ = [
    "TransformersBinaryRcAnnotator",
    "TransformersTokenClfAnnotator",
    "SpacyAnnotator",
    "EntityFishingAnnotator",
    "AllenNLPBinaryRcAnnotator"
]
