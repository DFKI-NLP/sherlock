import json
import os
import logging
import argparse

import utils
from relation_types import RELATION_TYPES


def map_fewrel_label(example):
    fewrel_label = example["label"]
    mapped_label = None

    if fewrel_label in ["capital of", "capital"]:
        mapped_label = "loc:capital_of"
    elif fewrel_label == "conflict":
        mapped_label = "per:conflict"
    elif fewrel_label in [
        "located in the administrative territorial entity",
        "located on terrain feature",
        # "located in or next to body of water"
    ]:
        mapped_label = "org:facility_or_location"
    elif fewrel_label == "conflict":
        mapped_label = "per:conflict"
    elif fewrel_label == "language":
        mapped_label = "per:language"
    # elif fewrel_label == "publisher":
    #     mapped_label = "publisher"  # TODO name, (per/org, loc) need NER to determine NER prefix for RE label
    elif fewrel_label == "location of formation":
        mapped_label = "org:location_of_formation"
    elif fewrel_label in ["head of government", "head of state"]:
        mapped_label = "gpe:head_of_gov/state"
    # elif fewrel_label == "location":
    #     mapped_label = "location"  # TODO name, (fac/event/item, loc) need NER to determine NER prefix for RE label
    elif fewrel_label == "country of citizenship":
        mapped_label = "per:country_of_citizenship"
    elif fewrel_label == "notable work":
        mapped_label = "per:notable_work"
    elif fewrel_label == "production company":
        mapped_label = "org:production_company"
    elif fewrel_label == "creator":
        mapped_label = "per:creator"
    elif fewrel_label == "ethnic group":
        mapped_label = "per:ethnic_group"
    elif fewrel_label in ["manufacturer", "product or material produced"]:
        mapped_label = "org:product_or_technology_or_service"
    elif fewrel_label == "position held":
        mapped_label = "per:title"
    elif fewrel_label == "producer":
        mapped_label = "per:producer"
    elif fewrel_label == "contains location":
        mapped_label = "loc:contains_location"
    elif fewrel_label == "author":
        mapped_label = "per:author"
    elif fewrel_label == "director":
        mapped_label = "per:director"
    elif fewrel_label == "work location":
        mapped_label = "per:work_location"
    elif fewrel_label == "religion":
        mapped_label = "per:religion"  # TODO political/religious_affiliation mapping?
    elif fewrel_label == "unemployment rate":
        mapped_label = "loc:unemployment_rate"
    elif fewrel_label == "country of origin":
        mapped_label = "loc:country_of_origin"
    elif fewrel_label == "performer":
        mapped_label = "per:performer"
    elif fewrel_label == "composer":
        mapped_label = "per:composer"
    elif fewrel_label == "lyrics by":
        mapped_label = "per:lyrics_by"
    elif fewrel_label == "director":
        mapped_label = "per:director"
    elif fewrel_label == "screenwriter":
        mapped_label = "per:screenwriter"
    elif fewrel_label == "developer":
        mapped_label = "per:developer"
    elif fewrel_label == "sister city":
        mapped_label = "loc:twinned_adm_body"

    elif fewrel_label in ["father", "mother"]:
        mapped_label = "per:parent"
    elif fewrel_label == "member of political party":
        mapped_label = "per:member_of_political_party"  # TODO (per, org) -> org:political/religious_affiliation
    elif fewrel_label == "hq location":
        mapped_label = "org:place_of_headquarters"
    elif fewrel_label == "sibling":
        mapped_label = "per:siblings"
    elif fewrel_label == "country":
        mapped_label = "loc:country"
    elif fewrel_label == "occupation":
        mapped_label = "per:title"
    elif fewrel_label == "residence":
        mapped_label = "per:places_of_residence"
    elif fewrel_label == "subsidiary":  # parent, subsidiary
        mapped_label = "org:subsidiaries"
    elif fewrel_label == "owned by":
        mapped_label = "org:owned_by"   # TODO parent company/shareholders?
    elif fewrel_label == "location of":
        mapped_label = "loc:location_of"
    elif fewrel_label == "field of work":
        mapped_label = "per:field_of_work"  # TODO check
    # TODO sort and check for missing mappings/arg positions

    if mapped_label is None:
        return None

    assert mapped_label in RELATION_TYPES
    example["label"] = mapped_label
    return example


def fewrel_converter(data, fewrel_rel_info):
    converted_examples = []
    for label, examples in data.items():
        for idx, example in enumerate(examples):
            head_token_positions = example["h"][2][0]
            tail_token_positions = example["t"][2][0]

            subj_start = head_token_positions[0]
            subj_end = head_token_positions[-1]
            obj_start = tail_token_positions[0]
            obj_end = tail_token_positions[-1]
            converted_example = map_fewrel_label({
                "id": "r/" + utils.generate_example_id(),
                "tokens": example["tokens"],
                "label": fewrel_rel_info[label][0],
                "grammar": ["SUBJ", "OBJ"],
                "entities": [[subj_start, subj_end+1], [obj_start, obj_end+1]],
                # "type": [subj_type, obj_type]
            })
            if converted_example is not None:
                converted_examples.append(converted_example)
    return converted_examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/fewrel",
        type=str,
        help="path to directory containing the fewrel relation info and data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/fewrel/converted",
        type=str,
        help="path to directory where the converted files should be saved",
    )
    args = parser.parse_args()

    fewrel_path = args.data_path
    export_path = args.export_path
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    rel_info_path = os.path.join(fewrel_path, "pid2name.json")
    logging.info("Reading doc relation info %s", rel_info_path)
    with open(rel_info_path, mode="r", encoding="utf-8") as f:
        fewrel_rel_info = json.load(f)

    for split in ["train", "val"]:
        split_path = os.path.join(fewrel_path, split + ".json")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        logging.info("Processing and exporting to %s", split_export_path)
        with open(split_path, mode="r", encoding="utf-8") as fewrel_file, \
                open(split_export_path, mode="w", encoding="utf-8") as fewrel_export_file:
            fewrel_data = json.load(fewrel_file)
            converted_examples = fewrel_converter(fewrel_data, fewrel_rel_info)
            for conv_example in converted_examples:
                fewrel_export_file.write(json.dumps(conv_example))
                fewrel_export_file.write("\n")


if __name__ == "__main__":
    main()
