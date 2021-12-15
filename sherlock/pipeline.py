import importlib
import json
import pkgutil
import sys
from typing import Any, List

import _jsonnet

from sherlock import Document, DocumentProcessor
from sherlock.annotators.annotator import Annotator


class Pipeline(DocumentProcessor):
    """
    A wrapper for a list of document processors.

    Optionally adds provenance information from the given pipeline config.
    """

    def __init__(self, processors: List[DocumentProcessor], provenance: List[Any] = None):
        self._processors: List[DocumentProcessor] = processors
        self._provenance = provenance

    def process_documents(self, documents: List[Document]) -> List[Document]:
        for processor in self._processors:
            processor.process_documents(documents)
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

        if "include_packages" in pipeline_config:
            for package_name in pipeline_config["include_packages"]:
                # Import any additional modules needed (to register custom classes)
                Pipeline._import_submodules(package_name)

        processors = []
        for params in pipeline_config["pipeline"]:
            # model_path = step["path"]
            # joint_path = os.path.normpath(
            #     os.path.join(os.path.dirname(args.pipeline_config), model_path)
            # )
            # if os.path.isdir(joint_path):
            #     model_path = joint_path
            params["device"] = cuda_device
            annotator_name = params.pop("name")
            processors.append(Annotator.by_name(annotator_name).from_pretrained(**params))
        provenance = pipeline_config.get("provenance", None)
        return Pipeline(processors=processors, provenance=provenance)

    @staticmethod
    def _import_submodules(package_name: str) -> None:
        """
        Import all submodules under the given package.
        Primarily useful to have custom classes get loaded and registered.
        """
        # Code taken from https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/common/util.py#L308
        importlib.invalidate_caches()

        # For some reason, python doesn't always add this by default to your path, but you pretty much
        # always want it when using `--include-package`.  And if it's already there, adding it again at
        # the end won't hurt anything.
        sys.path.append(".")

        # Import at top level
        module = importlib.import_module(package_name)
        path = getattr(module, "__path__", [])
        path_string = "" if not path else path[0]

        # walk_packages only finds immediate children, so need to recurse.
        for module_finder, name, _ in pkgutil.walk_packages(path):
            # Sometimes when you import third-party libraries that are on your path,
            # `pkgutil.walk_packages` returns those too, so we need to skip them.
            if path_string and module_finder.path != path_string:  # type: ignore[union-attr]
                continue
            subpackage = f"{package_name}.{name}"
            Pipeline._import_submodules(subpackage)
