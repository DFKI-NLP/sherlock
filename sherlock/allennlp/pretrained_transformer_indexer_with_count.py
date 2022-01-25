from typing import Dict, List, Optional, Tuple, Any
import logging
import torch
from allennlp.common.util import pad_sequence_to_length

from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer


logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained_transformer_with_count")
class PretrainedTransformerIndexerWithCount(PretrainedTransformerIndexer):
    """
    This `TokenIndexer` ist the same as `PretrainedTransformerIndxer` with
    the only difference that it is able to count vocabularies.
    # Parameters
    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = `None`)
        If not None, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerEmbedder`.
    tokenizer_kwargs : `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    """  # noqa: E501

    def __init__(
        self,
        # model_name: str,
        # namespace: str = "tags",
        # max_length: int = None,
        # tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # self._namespace = namespace
        # self._allennlp_tokenizer = PretrainedTransformerTokenizer(
        #     model_name, tokenizer_kwargs=tokenizer_kwargs
        # )
        # self._tokenizer = self._allennlp_tokenizer.tokenizer
        # self._added_to_vocabulary = False

        # self._num_added_start_tokens = len(self._allennlp_tokenizer.single_sequence_start_tokens)
        # self._num_added_end_tokens = len(self._allennlp_tokenizer.single_sequence_end_tokens)

        # self._max_length = max_length
        # if self._max_length is not None:
        #     num_added_tokens = len(self._allennlp_tokenizer.tokenize("a")) - 1
        #     self._effective_max_length = (  # we need to take into account special tokens
        #         self._max_length - num_added_tokens
        #     )
        #     if self._effective_max_length <= 0:
        #         raise ValueError(
        #             "max_length needs to be greater than the number of special tokens inserted."
        #         )

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if self._namespace is not None:
            text = self._get_feature_value(token)
            if self.lowercase_tokens:
                text = text.lower()
            counter[self.namespace][text] += 1

    @overrides
    def __eq__(self, other):
        if isinstance(other, PretrainedTransformerIndexerWithCount):
            for key in self.__dict__:
                if key == "_tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented
