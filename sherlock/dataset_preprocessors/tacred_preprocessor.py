import json
import os
import logging
import argparse
import re

import utils
from relation_types import RELATION_TYPES
from ner_types import NER_TYPES


def map_tacred_label(example, merge_location=True):
    tacred_label = example["label"]
    mapped_label = tacred_label

    if merge_location:
        # "org:city_of_headquarters", "org:country_of_headquarters", "org:stateorprovince_of_headquarters"
        # "per:city_of_birth", "per:country_of_birth", "per:stateorprovince_of_birth"
        # "per:cities_of_residence", "per:countries_of_residence", "per:stateorprovinces_of_residence"
        mapped_label = re.sub('(cities|countries|stateorprovinces)_of', 'places_of', tacred_label)
        mapped_label = re.sub('(city|country|stateorprovince)_of', 'place_of', mapped_label)

    assert mapped_label in RELATION_TYPES
    example["label"] = mapped_label
    return example


def map_tacred_ner_label(tacred_label):
    mapped_label = None
    if tacred_label == "PERSON":
        mapped_label = "PERSON"
    elif tacred_label == "ORGANIZATION":
        mapped_label = "ORG"
    elif tacred_label == "LOCATION":
        mapped_label = "LOC"
    elif tacred_label == "MISC":
        mapped_label = "MISC"
    elif tacred_label == "CITY":
        mapped_label = "LOC"
    elif tacred_label == "DATE":
        mapped_label = "DATE"
    elif tacred_label == "NATIONALITY":
        mapped_label = "LOC"    # TODO check this
    elif tacred_label == "RELIGION":
        mapped_label = "NORP"
    elif tacred_label == "URL":
        mapped_label = "URL"
    elif tacred_label == "CAUSE_OF_DEATH":
        mapped_label = "CAUSE_OF_DEATH"
    elif tacred_label == "COUNTRY":
        mapped_label = "LOC"
    elif tacred_label == "DURATION":
        mapped_label = "TIME"   # TODO check this
    elif tacred_label == "STATE_OR_PROVINCE":
        mapped_label = "LOC"
    elif tacred_label == "CRIMINAL_CHARGE":
        mapped_label = "CHARGE"
    elif tacred_label == "IDEOLOGY":
        mapped_label = "MISC"   # TODO check this
    elif tacred_label == "TITLE":
        mapped_label = "POSITION"
    if mapped_label is not None:
        assert mapped_label in NER_TYPES, f"{mapped_label} not valid label"
    return mapped_label


def tacred_converter(data):
    converted_examples = []
    for example in data:
        label = example["relation"]
        inverse = False
        entities = [[example["subj_start"], example["subj_end"]+1], [example["obj_start"], example["obj_end"]+1]]
        subj_type = example["subj_type"]
        obj_type = example["obj_type"]
        ent_type = [subj_type, obj_type]
        if inverse:
            entities[0], entities[1] = entities[1], entities[0]
            ent_type[0], ent_type[1] = ent_type[1], ent_type[0]
        converted_example = {
            "id": example["id"],
            "tokens": example["token"],
            "label": label,
            "grammar": ["SUBJ", "OBJ"],
            "entities": entities,
            "type": ent_type
        }
        converted_examples.append(map_tacred_label(converted_example))
    return converted_examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/tacred/data/json",
        type=str,
        help="path to directory containing the tacred data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/tacred/data/converted",
        type=str,
        help="path to directory where the converted files should be saved",
    )
    args = parser.parse_args()

    tacred_path = args.data_path
    export_path = args.export_path
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    for split in ["train", "dev", "test"]:
        split_path = os.path.join(tacred_path, split + ".json")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        logging.info("Processing and exporting to %s", split_export_path)
        with open(split_path, mode="r", encoding="utf-8") as tacred_file, \
                open(split_export_path, mode="w", encoding="utf-8") as tacred_export_file:
            tacred_data = json.load(tacred_file)
            logging.info(f"{len(tacred_data)} examples in original file")
            converted_examples = tacred_converter(tacred_data)
            logging.info(f"{len(converted_examples)} examples in converted file")
            for conv_example in converted_examples:
                tacred_export_file.write(json.dumps(conv_example))
                tacred_export_file.write("\n")
            logging.info(utils.get_label_counter(converted_examples))


if __name__ == "__main__":
    main()
