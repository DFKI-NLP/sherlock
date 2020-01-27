from typing import Any, Dict, List

import numpy as np

from sherlock.document import Document, Relation
from sherlock.predictors.predictor import Predictor


class BinaryRcPredictor(Predictor):
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
            doc = docs_by_guid[meta["guid"]]
            doc.rels.append(
                Relation(
                    doc=doc, head_idx=meta["head_idx"], tail_idx=meta["tail_idx"], label=prediction,
                )
            )

        return documents
