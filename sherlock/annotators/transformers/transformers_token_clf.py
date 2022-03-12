from typing import Any, Dict, List, Optional

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

from sherlock.document import Document, Mention
from sherlock.annotators.annotator import Annotator
from sherlock.annotators.transformers.transformers_annotator import TransformersAnnotator
from sherlock.tasks import NLPTask


@Annotator.register("transformers_token_clf")
class TransformersTokenClfAnnotator(TransformersAnnotator):
    task = NLPTask.TOKEN_CLASSIFICATION

    def combine(
        self,
        documents: List[Document],
        annotations: Optional[np.ndarray],
        label_ids: Optional[np.ndarray],
        metadata: List[Dict[str, Any]],
    ) -> List[Document]:
        # Return the documents as is if no annotations are included,
        # e.g. a single documents without entities.
        if annotations is None:
            return documents

        assert label_ids is not None, "Label ids are missing"
        annos = np.argmax(annotations, axis=2)

        labeled_annos: List[List[Any]] = [[] for _ in range(label_ids.shape[0])]
        for i in range(label_ids.shape[0]):
            for j in range(label_ids.shape[1]):
                if label_ids[i, j] != self.converter.pad_token_label_id:
                    labeled_annos[i].append(self.converter.id_to_label_map[annos[i][j]])

        for doc, labeled_annotation in zip(documents, labeled_annos):
            for label, start, end in get_entities(labeled_annotation):
                # end is inclusive, we want exclusive -> +1
                doc.ments.append(Mention(doc=doc, start=start, end=end + 1, label=label))

        return documents
