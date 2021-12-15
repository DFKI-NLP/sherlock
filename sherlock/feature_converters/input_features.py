import copy
import json
from typing import List, Dict, Union, Optional, Any

from allennlp.data.instance import Instance


class InputFeatures(object):
    """
    A single set of features.

    Parameters
    ----------
    metadata : ``Dict[str, Any]``, optional (default=`{}`)
        Dictionary mapping metadata keys to their values. Can
        be anything task specific, is used differently
        depending on the Annotator in use.
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]]=None,
    ) -> None:

        self.metadata = metadata or {}

    def __str__(self) -> str:
        return str(self.to_dict())

    def __repr__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> Dict:
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeaturesTransformers(InputFeatures):
    """
    A single set of InputFeatures for Transformers
    Parameters
    ----------
    input_ids : ``List[int]``
        Indices of input sequence tokens in the vocabulary.
    attention_mask : ``List[int]``, optional (default=`None`)
        Mask to avoid performing attention on padding token indices.
        Mask values selected in ``[0, 1]``:
        Usually  ``1`` for tokens that are NOT MASKED, ``0`` for
        MASKED (padded) tokens.
    token_type_ids : ``List[int]``, optional (default=`None`)
        Segment token indices to indicate first and second portions of the inputs.
    position_ids : ``List[int]``, optional (default=`None`)
        Identifiers for each token at which position it is, if None
        they are automatically created as absolute positional embeddings.
    head_mask : ``List[int]``, optional (default=`None`)
        Mask to nullify selected heads of the self-attention modules.
        Mask values selected in ``[0,1]``
    labels : ``int | List[int]``, optional (default=`None`)
        Labels corresponding to the input, can be a list or a single label.
    metadata : ``Dict[str, Any]``, optional (default=`{}`)
        Dictionary mapping metadata keys to their values. Can
        be anything task specific, is used differently
        depending on the Annotator in use.
    """

    def __init__(
        self,
        input_ids: List[int],
        attention_mask: Optional[List[int]] = None,
        token_type_ids: Optional[List[int]] = None,
        position_ids: Optional[List[int]] = None,
        head_mask: Optional[List[int]] = None,
        labels: Optional[Union[int, List[int]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(metadata)
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.head_mask = head_mask


class InputFeaturesAllennlp(InputFeatures):
    """
    A single set of features of data.
    Parameters
    ----------
    instance: ``allennlp.data.instance.Instance``
        allennlp Instance object containing all relevant fields to
        run predictions from a model.
    labels : ``int | List[int]``, optional (default=`None`)
        Labels corresponding to the input, can be a list or a single label.
    metadata: ``Dict[str,Any]``, optional (default=`{}`)
        Dictionary mapping metadata keys to their values. Can
        be anything task specific, is used differently
        depending on the Annotator in use.
    """

    def __init__(
        self,
        instance: Instance,
        labels: Optional[Union[int, List[int]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(metadata)
        self.instance = instance
        self.labels = labels
