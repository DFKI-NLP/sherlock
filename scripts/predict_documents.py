import argparse
import json
import logging
import math
import os
import sys
import torch
from time import time
from pathlib import Path

from sherlock.annotators.transformers.transformers_binary_rc import TransformersBinaryRcAnnotator
from sherlock.dataset_readers.dfki_tacred_jsonl import TacredDatasetReaderDfkiJsonl
from sherlock.dataset_preprocessors.utils import open_file


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


def predict_documents(input_path, model_path, output_path, document_batch_size, device=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_reader = TacredDatasetReaderDfkiJsonl()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    annotator = TransformersBinaryRcAnnotator.from_pretrained(model_path, device=device)

    batch_docs = []
    first_batch = True
    batch_counter = 1

    is_gzip = True if ".gz" in os.path.splitext(input_path)[1] else False
    num_of_examples = 0
    data = []
    with open_file(input_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if is_gzip:
                data.append(json.loads(line))
            num_of_examples += 1  # progress information
    if is_gzip:
        docs = dataset_reader._documents_generator(data)
    else:
        docs = dataset_reader.get_documents(file_path=input_path)

    num_batches = math.ceil(num_of_examples / document_batch_size)
    logger.info(f"Processing {num_of_examples} examples in {num_batches} batches")

    for doc_idx, doc in enumerate(docs):
        batch_docs.append(doc)
        if len(batch_docs) == document_batch_size or doc_idx + 1 == num_of_examples:
            logger.info(f"Processing batch {batch_counter}/{num_batches}")
            batch_start = time()
            annotated_docs = annotator.process_documents(batch_docs)
            write_mode = "w" if first_batch else "a"
            first_batch = False
            with open_file(output_path, mode=write_mode, encoding="utf-8") as out_file:
                for annotated_doc in annotated_docs:
                    out_file.write(json.dumps(annotated_doc.to_dict()) + "\n")
            batch_end = time()
            logger.info(f"Took {batch_end - batch_start} seconds for {len(batch_docs)} documents")
            batch_docs = []
            batch_counter += 1
    assert len(batch_docs) == 0, f"{len(batch_docs)} docs were not annotated"


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to input file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to trained transformers model",
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
    output_path = args.output_path
    document_batch_size = args.document_batch_size
    device = args.device

    predict_documents(input_path, model_path, output_path, document_batch_size, device=device)


if __name__ == "__main__":
    main()
