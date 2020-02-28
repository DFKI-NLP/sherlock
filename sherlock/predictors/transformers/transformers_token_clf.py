from typing import Any, Dict, List

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

from sherlock.document import Document, Mention
from sherlock.predictors import Predictor
from sherlock.predictors.transformers.transformers_predictor import TransformersPredictor
from sherlock.tasks import NLPTask


@Predictor.register("transformers_token_clf")
class TransformersTokenClfPredictor(TransformersPredictor):
    task = NLPTask.TOKEN_CLASSIFICATION

    def combine(
        self,
        documents: List[Document],
        predictions: np.array,
        label_ids: np.array,
        metadata: List[Dict[str, Any]],
    ) -> List[Document]:
        preds = np.argmax(predictions, axis=2)

        predictions = [[] for _ in range(label_ids.shape[0])]
        for i in range(label_ids.shape[0]):
            for j in range(label_ids.shape[1]):
                if label_ids[i, j] != self.converter.pad_token_label_id:
                    predictions[i].append(self.converter.id_to_label_map[preds[i][j]])

        for doc, prediction in zip(documents, predictions):
            for label, start, end in get_entities(prediction):
                # end is inclusive, we want exclusive -> +1
                doc.ments.append(Mention(doc=doc, start=start, end=end + 1, label=label))

        return documents
