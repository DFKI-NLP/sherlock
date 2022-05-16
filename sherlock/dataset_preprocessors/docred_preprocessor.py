import json
import os
import logging
import argparse

import utils
from relation_types import RELATION_TYPES


def map_doc_red_label(example):
    doc_red_label = example["label"]
    mapped_label = None

    if doc_red_label in ["capital of", "capital"]:
        mapped_label = "loc:capital_of"
    elif doc_red_label == "conflict":
        mapped_label = "per:conflict"
    elif doc_red_label in [
            "located in the administrative territorial entity",
            "located on terrain feature",
            "located in or next to body of water"
    ]:
        mapped_label = "loc:located_in"
    elif doc_red_label == "conflict":
        mapped_label = "per:conflict"
    elif doc_red_label == "language":
        mapped_label = "per:language"
    elif doc_red_label == "publisher":
        mapped_label = "publisher"  # TODO name
    elif doc_red_label == "location of formation":
        mapped_label = "org:location_of_formation"
    elif doc_red_label in ["head of government", "head of state"]:
        mapped_label = "head_of_gov/state"  # TODO name
    elif doc_red_label == "location":
        mapped_label = "location"   # TODO name
    elif doc_red_label == "country of citizenship":
        mapped_label = "per:country_of_citizenship"
    elif doc_red_label == "notable work":
        mapped_label = "per:notable_work"
    elif doc_red_label == "production company":
        mapped_label = "org:production_company"
    elif doc_red_label == "creator":
        mapped_label = "per:creator"
    elif doc_red_label == "ethnic group":
        mapped_label = "per:ethnic_group"
    elif doc_red_label in ["manufacturer", "product or material produced"]:
        mapped_label = "org:manufacturer"
    elif doc_red_label == "position held":
        mapped_label = "per:title"
    elif doc_red_label == "producer":
        mapped_label = "per:producer"
    elif doc_red_label == "contains location":
        mapped_label = "loc:contains_location"
    elif doc_red_label == "author":
        mapped_label = "per:author"
    elif doc_red_label == "director":
        mapped_label = "per:director"
    elif doc_red_label == "work location":
        mapped_label = "per:work_location"
    elif doc_red_label == "religion":
        mapped_label = "per:religion"   # TODO political/religious_affiliation mapping?
    elif doc_red_label == "unemployment rate":
        mapped_label = "loc:unemployment_rate"
    elif doc_red_label == "country of origin":
        mapped_label = "loc:country_of_origin"
    elif doc_red_label == "performer":
        mapped_label = "per:performer"
    elif doc_red_label == "composer":
        mapped_label = "per:composer"
    elif doc_red_label == "lyrics by":
        mapped_label = "per:lyrics_by"
    elif doc_red_label == "director":
        mapped_label = "per:director"
    elif doc_red_label == "screenwriter":
        mapped_label = "per:screenwriter"
    elif doc_red_label == "developer":
        mapped_label = "per:developer"
    elif doc_red_label == "sister city":
        mapped_label = "loc:twinned_adm_body"

    if mapped_label is None:
        return None

    assert mapped_label in RELATION_TYPES
    example["label"] = mapped_label
    return example


def doc_red_converter(example, docred_rel_info):
    labels = example["labels"]
    converted_examples = []
    for idx, label in enumerate(labels):
        rel_type = docred_rel_info[label["r"]]
        evidence = label["evidence"]
        head_idx = label["h"]
        tail_idx = label["t"]
        head_sent_ids = []
        tail_sent_ids = []
        for mention in example["vertexSet"][head_idx]:
            if mention["sent_id"] in evidence:
                head_sent_ids.append(mention["sent_id"])
        for mention in example["vertexSet"][tail_idx]:
            if mention["sent_id"] in evidence:
                tail_sent_ids.append(mention["sent_id"])
        common_sent_ids = list(set([sent_id for sent_id in head_sent_ids if sent_id in tail_sent_ids]))
        for sent_id in common_sent_ids:
            head = None
            for head_mention in example["vertexSet"][head_idx]:
                if head_mention["sent_id"] == sent_id:
                    head = head_mention
            tail = None
            for tail_mention in example["vertexSet"][tail_idx]:
                if tail_mention["sent_id"] == sent_id:
                    tail = tail_mention
            # subj/obj vs. head/tail
            subj_start = head["pos"][0]
            subj_end = head["pos"][1]
            obj_start = tail["pos"][0]
            obj_end = tail["pos"][1]
            # TODO NER mapping
            subj_type = head["type"]
            obj_type = tail["type"]
            converted_example = map_doc_red_label({
                "id": "r/" + utils.generate_example_id(),
                "tokens": example["sents"][sent_id],
                "label": rel_type,
                "grammar": ["SUBJ", "OBJ"],
                "entities": [[subj_start, subj_end], [obj_start, obj_end]],
                "type": [subj_type, obj_type]
            })
            if converted_example is not None:
                converted_examples.append(converted_example)
    return converted_examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/DocRED",
        type=str,
        help="path to directory containing the docred relation info and data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/DocRED/converted",
        type=str,
        help="path to directory where the converted files should be saved",
    )
    args = parser.parse_args()

    docred_path = args.data_path
    export_path = args.export_path
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    rel_info_path = os.path.join(docred_path, "rel_info.json")
    logging.info("Reading doc relation info %s", rel_info_path)
    with open(rel_info_path, mode="r", encoding="utf-8") as f:
        docred_rel_info = json.load(f)

    for split in ["train_annotated", "dev", "test"]:
        split_path = os.path.join(docred_path, split + ".json")
        logging.info("Reading %s", split_path)
        with open(split_path, mode="r", encoding="utf-8") as f:
            docred_data = json.load(f)

        split_export_path = os.path.join(export_path, split + ".jsonl")
        logging.info("Processing and exporting to %s", split_export_path)

        with open(split_export_path, mode="w", encoding="utf-8") as f:
            for example in docred_data:
                if "labels" in example:
                    converted_examples = doc_red_converter(example, docred_rel_info)
                    for conv_example in converted_examples:
                        f.write(json.dumps(conv_example))
                        f.write("\n")


if __name__ == "__main__":
    main()
