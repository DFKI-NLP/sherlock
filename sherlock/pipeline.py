import json
from typing import Any, List

import _jsonnet

from sherlock import Document, DocumentProcessor
from sherlock.predictors.predictor import Predictor


class Pipeline(DocumentProcessor):
    """
    A wrapper for a list of document processors.

    Optionally adds provenance information from the given pipeline config.
    """

    def __init__(self, processors: List[DocumentProcessor], provenance: List[Any] = None):
        self._processors: List[DocumentProcessor] = processors
        self._provenance = provenance

    def predict_documents(self, documents: List[Document]) -> List[Document]:
        for processor in self._processors:
            processor.predict_documents(documents)
        self._add_provenance(documents)
        return documents

    def _add_provenance(self, documents: List[Document]) -> None:
        if self._provenance is not None:
            for doc in documents:
                if doc.provenance is None:
                    doc.provenance = []
                doc.provenance.extend(self._provenance)

    @staticmethod
    def from_file(jsonnet_path: str, cuda_device: Any = -1):
        pipeline_config = json.loads(_jsonnet.evaluate_file(jsonnet_path))
        cuda_device = cuda_device if cuda_device >= 0 else "cpu"
        processors = []
        for params in pipeline_config["pipeline"]:
            # model_path = step["path"]
            # joint_path = os.path.normpath(
            #     os.path.join(os.path.dirname(args.pipeline_config), model_path)
            # )
            # if os.path.isdir(joint_path):
            #     model_path = joint_path
            params["device"] = cuda_device
            predictor_name = params.pop("name")
            processors.append(Predictor.by_name(predictor_name).from_pretrained(**params))
        provenance = pipeline_config.get("provenance", None)
        return Pipeline(processors=processors, provenance=provenance)
