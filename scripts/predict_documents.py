import argparse
import json
import logging
import math
import sys

import torch

from time import time
from pathlib import Path

from sherlock.annotators.transformers.transformers_binary_rc import TransformersBinaryRcAnnotator
from sherlock.dataset_readers.dfki_tacred_jsonl import TacredDatasetReaderDfkiJsonl
from sherlock.dataset_preprocessors.utils import open_file, get_entities, _compute_majority_tag
from sherlock.document import Document


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


def get_businesswire_documents(data_path):
    # Sets "ent_type" for each token to the majority vote label (model ensemble output) and fill "ments" field
    data_path = Path(data_path)
    with open_file(data_path, mode="r") as input_file:
        businesswire_data = (json.loads(line) for line in input_file)
        for document in businesswire_data:
            ner_labels = []
            for idx, token in enumerate(document["tokens"]):
                if token["ent_type"]:
                    majority_ner_label = token["ent_type"]
                else:
                    majority_ner_label, _ = _compute_majority_tag(token)
                    token.pop("ent_dist")
                    document["tokens"][idx]["ent_type"] = majority_ner_label
                ner_labels.append(majority_ner_label)
            if document["ments"] is None or len(document["ments"]) == 0:
                document["ments"] = get_entities(ner_labels)
            yield Document.from_dict(document)


def predict_documents(docs, model_path, output_path, document_batch_size, num_of_examples, no_cuda=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    annotator = TransformersBinaryRcAnnotator.from_pretrained(model_path, device=device)

    batch_docs = []
    first_batch = True
    batch_counter = 1

    if document_batch_size is None:
        document_batch_size = num_of_examples
    num_batches = math.ceil(num_of_examples / document_batch_size)
    logger.info(f"Processing {num_of_examples} documents in {num_batches} batches")
    start = time()
    processed_doc_guid = []

    for doc_idx, doc in enumerate(docs):
        batch_docs.append(doc)
        if len(batch_docs) == document_batch_size or doc_idx + 1 == num_of_examples:
            logger.info(f"Processing batch {batch_counter}/{num_batches}")
            batch_start = time()
            annotated_docs = annotator.process_documents(batch_docs)
            # assert len(batch_docs) == len(annotated_docs), \
            #     f"Mismatch: {len(batch_docs)} input docs vs. {len(annotated_docs)}"
            write_mode = "w" if first_batch else "a"
            first_batch = False
            with open_file(output_path, mode=write_mode, encoding="utf-8") as out_file:
                for annotated_doc in annotated_docs:
                    if annotated_doc.guid in processed_doc_guid:
                        continue
                    processed_doc_guid.append(annotated_doc.guid)
                    out_file.write(json.dumps(annotated_doc.to_dict()) + "\n")
            batch_end = time()
            logger.info(f"Took {batch_end - batch_start} seconds for {len(batch_docs)} documents")
            batch_docs = []
            batch_counter += 1
    end = time()
    logger.info(f"Took {end - start} seconds for {num_of_examples} documents")
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
        default=100,
        help="Batch size for document processing"
    )
    parser.add_argument(
        "--businesswire_prediction",
        action="store_true",
        default=False,
        help="Create documents generator directly from data"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        default=False,
        help="Avoid using CUDA when available"
    )
    args = parser.parse_args()
    input_path = args.input_path
    model_path = args.model_path
    output_path = args.output_path
    document_batch_size = args.document_batch_size
    businesswire_prediction = args.businesswire_prediction
    no_cuda = args.no_cuda

    num_of_examples = 0
    with open_file(input_path, "r", encoding="utf-8") as input_file:
        for _ in input_file:
            num_of_examples += 1  # progress information
    if businesswire_prediction:
        docs = get_businesswire_documents(data_path=input_path)
    else:
        dataset_reader = TacredDatasetReaderDfkiJsonl()
        docs = dataset_reader.get_documents(file_path=input_path)

    predict_documents(docs, model_path, output_path, document_batch_size,
                      num_of_examples=num_of_examples, no_cuda=no_cuda)


if __name__ == "__main__":
    main()
