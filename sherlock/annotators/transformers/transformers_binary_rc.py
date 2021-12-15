from typing import Any, Dict, List, Optional

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from sherlock.document import Document, Relation
from sherlock.feature_converters import FeatureConverter
from sherlock.annotators.annotator import Annotator
from sherlock.annotators.transformers.transformers_annotator import TransformersAnnotator
from sherlock.tasks import NLPTask


@Annotator.register("transformers_binary_rc")
class TransformersBinaryRcAnnotator(TransformersAnnotator):
    task = NLPTask.SEQUENCE_CLASSIFICATION

    def __init__(
        self,
        converter: FeatureConverter,
        model: PreTrainedModel,
        device: str = "cpu",
        batch_size: int = 16,
        ignore_no_relation: bool = True,
        add_logits: bool = False,
    ) -> None:
        super().__init__(converter, model, device, batch_size)
        self.ignore_no_relation = ignore_no_relation
        self.add_logits = add_logits

    def combine(
        self,
        documents: List[Document],
        annotations: Optional[np.ndarray],
        label_ids: Optional[np.ndarray],
        metadata: List[Dict[str, Any]],
    ) -> List[Document]:
        docs_by_guid = {doc.guid: doc for doc in documents}

        # annotations can be None, e.g. if less than two entity mentions were found in the document
        if annotations is not None:
            annotated_label_idxs = np.argmax(annotations, axis=1)

            for annotation_idx in range(len(annotations)):
                annotated_label_idx = annotated_label_idxs[annotation_idx]
                annotated_label = self.converter.id_to_label_map[annotated_label_idx]

                if self.ignore_no_relation and annotated_label == "no_relation":
                    continue

                named_logits: Optional[Dict[str, float]] = None
                if self.add_logits:
                    logits = annotations[annotation_idx, :].tolist()
                    named_logits = {
                        self.converter.id_to_label_map[logit_idx]: logit
                        for logit_idx, logit in enumerate(logits)
                    }

                meta = metadata[annotation_idx]
                doc = docs_by_guid[meta["guid"]]
                doc.rels.append(
                    Relation(
                        doc=doc,
                        head_idx=meta["head_idx"],
                        tail_idx=meta["tail_idx"],
                        label=annotated_label,
                        logits=named_logits,
                    )
                )

        return documents
