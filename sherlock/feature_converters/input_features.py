import copy
import json
from typing import List, Dict, Union
from allennlp.data.token_indexers.token_indexer import IndexedTokenList

from allennlp.data.tokenizers.token_class import Token


class InputFeatures(object):
    """
    A single set of features
    """

    def __init__(
        self,
        input_ids: Union[IndexedTokenList,any],
        labels: Union[int, List[int]]=None,
        metadata: Dict[str, any]=None,
    ) -> None:

        self.input_ids = input_ids
        self.labels = labels
        self.metadata = metadata or {}

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



class InputFeaturesTransformer(InputFeatures):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
            (TODO: allennlp indicates otherwise: https://github.com/allenai/allennlp/blob/c557d512edb6200dca15139316475c5b42432660/allennlp/data/tokenizers/pretrained_transformer_tokenizer.py#L263)
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    # TODO: types!
    def __init__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels: Union[int, List[int]]=None,
        metadata: Dict[str, any]=None,
    ) -> None:

        super().__init__(input_ids, labels, metadata)
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.head_mask = head_mask



class InputFeaturesAllennlp(InputFeatures):
    """
    A single set of features of data.
    Args:
        tokens: Indices of input sequence tokens in the vocabulary.
        labels: single integer or list of integers indicating labels
        metadata: Metadata about the InputFeatures.
    """

    def __init__(
        self,
        tokens: List[Token],
        input_ids: IndexedTokenList,
        labels: Union[int, List[int]]=None,
        metadata: Dict[str, any]=None,
    ) -> None:
        super().__init__(input_ids, labels, metadata)
        self.tokens = tokens
