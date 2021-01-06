from typing import Any, Dict, List, Optional

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from sherlock.document import Document, Relation
from sherlock.feature_converters import FeatureConverter
from sherlock.predictors.predictor import Predictor
from sherlock.predictors.transformers.transformers_predictor import TransformersPredictor
from sherlock.tasks import NLPTask


@Predictor.register("transformers_binary_rc")
class TransformersBinaryRcPredictor(TransformersPredictor):
    task = NLPTask.SEQUENCE_CLASSIFICATION

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        converter: FeatureConverter,
        model: PreTrainedModel,
        device: str = "cpu",
        batch_size: int = 16,
        ignore_no_relation: bool = True,
        add_logits: bool = False,
    ) -> None:
        super().__init__(tokenizer, converter, model, device, batch_size)
        self.ignore_no_relation = ignore_no_relation
        self.add_logits = add_logits

    def combine(
        self,
        documents: List[Document],
        predictions: Optional[np.array],
        label_ids: Optional[np.array],
        metadata: List[Dict[str, Any]],
    ) -> List[Document]:
        docs_by_guid = {doc.guid: doc for doc in documents}

        # predictions can be None, e.g. if less than two entity mentions were found in the document
        if predictions is not None:
            predicted_label_idxs = np.argmax(predictions, axis=1)

            for prediction_idx in range(len(predictions)):
                predicted_label_idx = predicted_label_idxs[prediction_idx]
                predicted_label = self.converter.id_to_label_map[predicted_label_idx]

                if self.ignore_no_relation and predicted_label == "no_relation":
                    continue

                named_logits: Optional[Dict[str, float]] = None
                if self.add_logits:
                    logits = predictions[prediction_idx, :].tolist()
                    named_logits = {
                        self.converter.id_to_label_map[logit_idx]: logit
                        for logit_idx, logit in enumerate(logits)
                    }

                meta = metadata[prediction_idx]
                doc = docs_by_guid[meta["guid"]]
                doc.rels.append(
                    Relation(
                        doc=doc,
                        head_idx=meta["head_idx"],
                        tail_idx=meta["tail_idx"],
                        label=predicted_label,
                        logits=named_logits,
                    )
                )

        return documents
