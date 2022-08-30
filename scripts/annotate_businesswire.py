import argparse
import json
import logging
import os
import sys
from time import time
from pathlib import Path

from sherlock.dataset_preprocessors.utils import open_file
from sherlock.dataset_preprocessors.businesswire_preprocessor import process_businesswire
from predict_documents import predict_documents


# Setup logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def annotate_doc(original_doc, doc_annotated_examples):
    return original_doc


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to businesswire file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to trained transformers model",
    )
    parser.add_argument(
        "--temp_file_path",
        type=str,
        help="Path to store temporary converted businesswire file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output file",
    )
    parser.add_argument(
        "--document_batch_size",
        type=int,
        help="Batch size for document processing",
        default=1000,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Cuda device"
    )
    args = parser.parse_args()
    input_path = args.input_path
    model_path = args.model_path
    temp_file_path = Path(args.temp_file_path)
    output_path = Path(args.output_path)
    document_batch_size = args.document_batch_size
    device = args.device

    # 1. Convert the businesswire to the dfki tacred jsonl format
    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
    process_businesswire(data_path=input_path, export_path=temp_file_path)

    # 2. Predict relations
    predict_documents(input_path=temp_file_path, model_path=model_path,
                      output_path=output_path, document_batch_size=document_batch_size, device=device)

    # 3. Add relations to businesswire data in the original format (assumes that the document order was preserved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_file(input_path, mode="r") as original_file, open_file(temp_file_path, mode="r") as annotation_file, \
            open_file(output_path, mode="w") as export_file:
        annotated_line = annotation_file.readline()
        for line in original_file:
            doc_annotated_examples = []
            original_doc = json.loads(line)
            same_doc = True
            while same_doc:
                if annotated_line:
                    annotated_example = json.loads(annotated_line)
                    if annotated_example["guid"] == original_doc["guid"]:
                        doc_annotated_examples.append(annotated_example)
                    else:
                        same_doc = False
                else:
                    same_doc = False
                annotated_line = annotation_file.readline()
            annotated_doc = annotate_doc(original_doc, doc_annotated_examples)
            if annotated_doc:
                export_file.write(json.dumps(annotated_doc) + "\n")


if __name__ == "__main__":
    main()
