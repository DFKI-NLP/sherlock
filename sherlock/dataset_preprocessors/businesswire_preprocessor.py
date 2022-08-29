import argparse
import json
import logging
import os
from pathlib import Path
from typing import List
from itertools import permutations
from allennlp.data.dataset_readers.dataset_utils import span_utils

from utils import open_file


def get_entities(ner_labels: List[str], tagging_format: str = "bio") -> List[dict]:
    """
    Given a sequence corresponding to e.g. BIO tags, extracts named entities.

    Parameters
    ----------
    ner_labels : List[str]
        Sequence of NER tags
    tagging_format : str, default="bio"
        Used to determine which span util function to use

    Returns
    ----------
    entities : List[dict]
        List of entity dictionaries with spans and entity label
    """
    assert tagging_format in [
        "bio",
        "iob1",
        "bioul",
    ], "Valid tagging format options are ['bio', 'iob1', 'bioul']"
    if tagging_format == "iob1":
        tags_to_spans = span_utils.iob1_tags_to_spans
    elif tagging_format == "bioul":
        tags_to_spans = span_utils.bioul_tags_to_spans
    else:
        tags_to_spans = span_utils.bio_tags_to_spans

    typed_string_spans = tags_to_spans(ner_labels)
    entities = []
    for label, span in typed_string_spans:
        entities.append(
            {
                "start": span[0],
                "end": span[1] + 1,  # make span exclusive
                "label": label,
            }
        )
    entities.sort(key=lambda e: e["start"])
    return entities


def businesswire_converter(data):
    for document in data:
        guid = document["guid"]
        for sent_idx, span in enumerate(document["sents"]):
            ner_labels = []
            tokens = []
            annotated_tokens = [token for token in document["tokens"][span["start"]: span["end"]]]
            for token in annotated_tokens:
                if token["ent_type"]:
                    ner_labels.append(token["ent_type"])
                else:
                    ent_dist = token["ent_dist"]
                    majority_ner_label = max(ent_dist, key=ent_dist.get)
                    ner_labels.append(majority_ner_label)
                tokens.append(document["text"][token["start"]:token["end"]])
            entities = get_entities(ner_labels)

            entity_pairs = permutations(entities, r=2)
            # TODO exclude pairs with impossible ent type combination (?)
            for ent_pair in entity_pairs:
                subj = ent_pair[0]
                obj = ent_pair[1]

                converted_example = {
                    "id": guid,
                    "sent_idx": sent_idx,   # needed to reconstruct documents after the prediction step
                    "tokens": tokens,
                    "label": None,
                    "grammar": ["SUBJ", "OBJ"],
                    "entities": [[subj["start"], subj["end"]], [obj["start"], obj["end"]]],
                    "type": [subj["label"], obj["label"]]
                }
                yield converted_example


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/businesswire/businesswire_20210115_20210912_v2_dedup_token_assembled.jsonlines.gz",
        type=str,
        help="path to businesswire file",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/businesswire/converted/businesswire_20210115_20210912_v2_dedup_token_assembled"
                ".jsonlines.gz",
        type=str,
        help="path to save converted businesswire file",
    )
    args = parser.parse_args()

    data_path = args.data_path
    is_gzip = True if ".gz" in os.path.splitext(data_path)[1] else False
    data_path = Path(data_path)
    export_path = Path(args.export_path)
    export_path.parent.absolute().mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logging.info("Reading %s", data_path)
    with open_file(data_path, mode="r", encoding="utf-8", is_gzip=is_gzip) as input_file, \
            open_file(export_path, mode="w", encoding="utf-8", is_gzip=is_gzip) as output_file:
        businesswire_data = (json.loads(line) for line in input_file)

        logging.info("Processing and exporting to %s", export_path)
        converted_examples = businesswire_converter(businesswire_data)

        example_counter = 0
        for conv_example in converted_examples:
            output_file.write(json.dumps(conv_example))
            output_file.write("\n")
            example_counter += 1

        logging.info(f"{example_counter} examples in converted file")


if __name__ == "__main__":
    main()
