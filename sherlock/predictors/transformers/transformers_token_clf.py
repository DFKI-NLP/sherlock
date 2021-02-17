from typing import Any, Dict, List, Optional

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

from sherlock.document import Document, Mention
from sherlock.predictors.predictor import Predictor
from sherlock.predictors.transformers.transformers_predictor import TransformersPredictor
from sherlock.tasks import NLPTask


@Predictor.register("transformers_token_clf")
class TransformersTokenClfPredictor(TransformersPredictor):
    task = NLPTask.TOKEN_CLASSIFICATION

    def combine(
        self,
        documents: List[Document],
        predictions: Optional[np.ndarray],
        label_ids: Optional[np.ndarray],
        metadata: List[Dict[str, Any]],
    ) -> List[Document]:
        # Return the documents as is if no predictions are included,
        # e.g. a single documents without entities.
        if predictions is None:
            return documents

        assert label_ids is not None, "Label ids are missing"
        preds = np.argmax(predictions, axis=2)

        labeled_preds: List[List[Any]] = [[] for _ in range(label_ids.shape[0])]
        for i in range(label_ids.shape[0]):
            for j in range(label_ids.shape[1]):
                if label_ids[i, j] != self.converter.pad_token_label_id:
                    labeled_preds[i].append(self.converter.id_to_label_map[preds[i][j]])

        for doc, labeled_prediction in zip(documents, labeled_preds):
            for label, start, end in get_entities(labeled_prediction):
                # end is inclusive, we want exclusive -> +1
                doc.ments.append(Mention(doc=doc, start=start, end=end + 1, label=label))

        return documents
