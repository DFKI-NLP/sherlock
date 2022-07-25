import json
import os
import logging
import argparse

import utils
from relation_types import RELATION_TYPES
from ner_types import NER_TYPES


def map_kbp37_label(example, override_entity_types=True):
    kbp37_label = example["label"]
    mapped_label = None
    if "type" in example:
        subj_type, obj_type = example["type"]
    else:
        subj_type, obj_type = None, None
    original_types = subj_type, obj_type

    if kbp37_label == "no_relation":
        mapped_label = kbp37_label
    elif kbp37_label == "org:alternate_names(e1,e2)":   # (org, alternate name)
        mapped_label = "org:alternate_names"
    elif kbp37_label == "rg:alternate_names(e2,e1)":   # (alternate name, org)
        mapped_label = "org:alternate_names"
        example = utils.swap_args(example)
    elif kbp37_label == "org:city_of_headquarters(e1,e2)":   # (org, hq city)
        mapped_label = "org:place_of_headquarters"
    elif kbp37_label == "org:city_of_headquarters(e2,e1)":   # (hq city, org)
        mapped_label = "org:place_of_headquarters"
        example = utils.swap_args(example)
    elif kbp37_label == "org:country_of_headquarters(e1,e2)":   # (org, hq country)
        mapped_label = "org:place_of_headquarters"
    elif kbp37_label == "org:country_of_headquarters(e2,e1)":   # (hq country, org)
        mapped_label = "org:place_of_headquarters"
        example = utils.swap_args(example)
    elif kbp37_label == "org:founded(e1,e2)":   # (org, date)
        mapped_label = "org:founded"
    elif kbp37_label == "org:founded(e2,e1)":   # (date, org)
        mapped_label = "org:founded"
        example = utils.swap_args(example)
    elif kbp37_label == "org:founded_by(e1,e2)":   # (org, founder)
        mapped_label = "org:founded_by"
    elif kbp37_label == "org:founded_by(e2,e1)":   # (founder, org)
        mapped_label = "org:founded_by"
        example = utils.swap_args(example)
    elif kbp37_label == "org:members(e1,e2)":   # (org, member)
        mapped_label = "org:members"
    elif kbp37_label == "org:members(e2,e1)":   # (member, org)
        mapped_label = "org:members"
        example = utils.swap_args(example)
    elif kbp37_label == "org:stateorprovince_of_headquarters(e1,e2)":   # (org, state)
        mapped_label = "org:place_of_headquarters"
    elif kbp37_label == "org:stateorprovince_of_headquarters(e2,e1)":   # (state, org)
        mapped_label = "org:place_of_headquarters"
        example = utils.swap_args(example)
    elif kbp37_label == "org:subsidiaries(e1,e2)":   # (org, subsidiary)
        mapped_label = "org:subsidiaries"
    elif kbp37_label == "org:subsidiaries(e2,e1)":   # (subsidiary, org)
        mapped_label = "org:subsidiaries"
        example = utils.swap_args(example)
    elif kbp37_label == "org:top_members/employees(e1,e2)":   # (org, employee)
        mapped_label = "org:top_members/employees"
    elif kbp37_label == "org:top_members/employees(e2,e1)":   # (employee, org)
        mapped_label = "org:top_members/employees"
        example = utils.swap_args(example)
    elif kbp37_label == "per:alternate_names(e1,e2)":   # (per, alternate name)
        mapped_label = "per:alternate_names"
    elif kbp37_label == "per:alternate_names(e2,e1)":   # (alternate name, per)
        mapped_label = "per:alternate_names"
        example = utils.swap_args(example)
    elif kbp37_label == "per:cities_of_residence(e1,e2)":   # (per, city)
        mapped_label = "per:places_of_residence"
    elif kbp37_label == "per:cities_of_residence(e2,e1)":   # (city, per)
        mapped_label = "per:places_of_residence"
        example = utils.swap_args(example)
    elif kbp37_label == "per:countries_of_residence(e1,e2)":   # (per, country)
        mapped_label = "per:places_of_residence"
    elif kbp37_label == "per:countries_of_residence(e2,e1)":   # (country, per)
        mapped_label = "per:places_of_residence"
        example = utils.swap_args(example)
    elif kbp37_label == "per:country_of_birth(e1,e2)":   # (per, country)
        mapped_label = "per:place_of_birth"
    elif kbp37_label == "per:country_of_birth(e2,e1)":   # (country, per)
        mapped_label = "per:place_of_birth"
        example = utils.swap_args(example)
    elif kbp37_label == "per:employee_of(e1,e2)":   # (per, org)
        mapped_label = "per:employee_of"
    elif kbp37_label == "per:employee_of(e2,e1)":   # (org, per)
        mapped_label = "per:employee_of"
        example = utils.swap_args(example)
    elif kbp37_label == "per:origin(e1,e2)":   # (per, loc)
        mapped_label = "per:origin"
    elif kbp37_label == "per:origin(e2,e1)":   # (loc, per)
        mapped_label = "per:origin"
        example = utils.swap_args(example)
    elif kbp37_label == "per:spouse(e1,e2)":   # (per, per)
        mapped_label = "per:spouse"
    elif kbp37_label == "per:spouse(e2,e1)":   # (per, per)
        mapped_label = "per:spouse"
        example = utils.swap_args(example)
    elif kbp37_label == "per:stateorprovinces_of_residence(e1,e2)":   # (per, state)
        mapped_label = "per:places_of_residence"
    elif kbp37_label == "per:stateorprovinces_of_residence(e2,e1)":   # (state, per)
        mapped_label = "per:places_of_residence"
        example = utils.swap_args(example)
    elif kbp37_label == "per:title(e1,e2)":   # (per, title)
        mapped_label = "per:title"
    elif kbp37_label == "per:title(e2,e1)":   # (title, per)
        mapped_label = "per:title"
        example = utils.swap_args(example)

    if mapped_label is None:
        return None

    assert mapped_label in RELATION_TYPES
    example["label"] = mapped_label
    if "type" in example:
        if not override_entity_types:
            subj_type, obj_type = original_types
        example["type"] = [map_kbp37_ner_label(subj_type), map_kbp37_ner_label(obj_type)]
    return example


def map_kbp37_ner_label(kbp37_label):
    # not really necessary since we either do not include NER labels or use correct NER labels from plass ner model
    mapped_label = kbp37_label
    # if kbp37_label == "PER":
    #     mapped_label = "PERSON"
    # elif kbp37_label == "ORG":
    #     mapped_label = "ORG"

    assert mapped_label in NER_TYPES, f"{mapped_label} not valid label"
    return mapped_label


def kbp37_converter(data, return_num_discarded=False, spacy_ner_predictor=None):
    num_discarded = 0
    converted_examples = []
    for example in data:
        text = example["example"][1]

        text = text.replace("<e1>", " <e1> ")
        text = text.replace("<e2>", " <e2> ")
        text = text.replace("</e1>", " </e1> ")
        text = text.replace("</e2>", " </e2> ")
        text = text.strip().replace(r"\s\s+", r"\s")
        tokens = text.split()

        subj_start = tokens.index("<e1>")
        obj_start = tokens.index("<e2>")
        if subj_start < obj_start:
            tokens.pop(subj_start)
            subj_end = tokens.index("</e1>")
            tokens.pop(subj_end)
            obj_start = tokens.index("<e2>")
            tokens.pop(obj_start)
            obj_end = tokens.index("</e2>")
            tokens.pop(obj_end)
        else:
            tokens.pop(obj_start)
            obj_end = tokens.index("</e2>")
            tokens.pop(obj_end)
            subj_start = tokens.index("<e1>")
            tokens.pop(subj_start)
            subj_end = tokens.index("</e1>")
            tokens.pop(subj_end)

        converted_example = {
            "id": example["example"][0],
            "tokens": tokens,
            "label": example["relation"],
            "grammar": ["SUBJ", "OBJ"],
            "entities": [[subj_start, subj_end], [obj_start, obj_end]]
        }
        if spacy_ner_predictor is not None:
            doc = spacy_ner_predictor(tokens)
            subj_type = utils.get_entity_type(doc, subj_start, subj_end)
            obj_type = utils.get_entity_type(doc, obj_start, obj_end)
            converted_example["type"] = [subj_type, obj_type]
        converted_example = map_kbp37_label(converted_example)
        if converted_example is not None:
            converted_examples.append(converted_example)
        else:
            num_discarded += 1
    if return_num_discarded:
        return converted_examples, num_discarded
    else:
        return converted_examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/kbp37",
        type=str,
        help="path to directory containing the kbp37 data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/kbp37/converted",
        type=str,
        help="path to directory where the converted files should be saved",
    )
    parser.add_argument(
        "--ner_model_path",
        # default="./models/spacy_trf/model-best",
        type=str,
        help="path to ner model",
    )
    args = parser.parse_args()

    kbp37_path = args.data_path
    export_path = args.export_path
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    spacy_ner_predictor = utils.load_spacy_predictor(args.ner_model_path) if args.ner_model_path else None

    for split in ["train", "dev", "test"]:
        split_path = os.path.join(kbp37_path, split + ".txt")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        kbp37_examples = []
        with open(split_path, mode="r", encoding="utf-8") as kbp37_file:
            example_line = None
            for idx, line in enumerate(kbp37_file.readlines()):
                line_no = idx % 4   # first line contains example, second line relation, third and fourth lines are \n
                if line_no == 0:
                    example_line = line.strip().split("\t")
                elif line_no == 1:
                    kbp37_examples.append({"example": example_line, "relation": line.strip()})
            logging.info(f"{len(kbp37_examples)} examples in original file")

        logging.info("Processing and exporting to %s", split_export_path)
        converted_examples, num_discarded = kbp37_converter(kbp37_examples,
                                                             return_num_discarded=True,
                                                             spacy_ner_predictor=spacy_ner_predictor)

        logging.info(f"{num_discarded} examples were discarded during label mapping")

        final_examples = []
        for example in converted_examples:
            if "type" in example and "O" in example["type"]:
                logging.warning(f"Examples has erroneous entity types: [{example}]")
            else:
                final_examples.append(converted_examples)
        logging.info(f"Removed {len(converted_examples)-len(final_examples)} examples with erroneous entity types")
        logging.info(f"{len(final_examples)} examples in converted file")

        with open(split_export_path, mode="w", encoding="utf-8") as export_kbp37_file:
            for converted_example in final_examples:
                export_kbp37_file.write(json.dumps(converted_example))
                export_kbp37_file.write("\n")
        logging.info(utils.get_label_counter(final_examples))


if __name__ == "__main__":
    main()
