from typing import List

import copy
import json

from transformers import PreTrainedTokenizer
from sherlock import Document


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self,
                 input_ids,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 labels=None,
                 metadata=None) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.head_mask = head_mask
        self.labels = labels
        self.metadata = metadata

    def __str__(self) -> str:
        return self.to_dict()

    def __repr__(self) -> str:
        return str(self.to_dict())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class FeatureConverter:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 labels: List[str],
                 max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        self.label_map = {l: i for i, l in enumerate(labels)}

    def document_to_features(self, document: Document,
                             verbose: bool = False) -> List[InputFeatures]:
        raise NotImplementedError("FeatureConvert must implement 'document_to_features'.")

    def documents_to_features(self, documents: List[Document]) -> List[InputFeatures]:
        input_features = []
        for document in documents:
            input_features.extend(self.document_to_features(document))
        return input_features
