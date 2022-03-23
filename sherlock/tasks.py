# -*- coding: utf8 -*-
from enum import Enum

from transformers import (
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
)

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

class IETask(Enum):
    NER = "ner"
    BINARY_RC = "binary_rc"


class NLPTask(Enum):
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    NONE = "none"


NLP_TASK_CLASSES = {
    NLPTask.SEQUENCE_CLASSIFICATION: {
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
        "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
        "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    },
    NLPTask.TOKEN_CLASSIFICATION: {
        "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
        "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
        "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
        "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
    },
}

ALLENNLP_TASK_CLASSES = {
    NLPTask.SEQUENCE_CLASSIFICATION: {
        "bert": ("", BertForSequenceClassification, PretrainedTransformerTokenizer, PretrainedTransformerIndexer),
    },
    NLPTask.TOKEN_CLASSIFICATION: {
        "bert": ("", BertForTokenClassification, PretrainedTransformerTokenizer, PretrainedTransformerIndexer),
    }
}