from typing import Any, Dict, List

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from sherlock.document import Document, Relation
from sherlock.feature_converters import FeatureConverter
from sherlock.predictors.predictor import Predictor
from sherlock.tasks import NLPTask


@Predictor.register("binary_rc")
class BinaryRcPredictor(Predictor):
    name = "binary_rc"
    task = NLPTask.SEQUENCE_CLASSIFICATION

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        converter: FeatureConverter,
        model: PreTrainedModel,
        device: str = "cpu",
        batch_size: int = 16,
        ignore_no_relation: bool = True,
    ) -> None:
        super().__init__(tokenizer, converter, model, device, batch_size)
        self.ignore_no_relation = ignore_no_relation

    def combine(
        self,
        documents: List[Document],
        predictions: np.array,
        label_ids: np.array,
        metadata: List[Dict[str, Any]],
    ) -> List[Document]:
        preds = np.argmax(predictions, axis=1)
        # probabilities = np.array([preds[i, j] for i, j in enumerate(np.argmax(preds, axis=1))])
        predictions = [self.converter.id_to_label_map[p] for p in preds]

        docs_by_guid = {doc.guid: doc for doc in documents}

        for prediction, meta in zip(predictions, metadata):
            if self.ignore_no_relation and prediction == "no_relation":
                continue

            doc = docs_by_guid[meta["guid"]]
            doc.rels.append(
                Relation(
                    doc=doc, head_idx=meta["head_idx"], tail_idx=meta["tail_idx"], label=prediction,
                )
            )

        return documents
