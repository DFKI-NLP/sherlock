import json
import csv
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

    if kbp37_label == "org:alternate_names(e1,e2)":   #
        mapped_label = "org:alternate_names"
    elif kbp37_label == "rg:alternate_names(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "org:city_of_headquarters(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "org:city_of_headquarters(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "org:country_of_headquarters(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "org:country_of_headquarters(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "org:founded(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "org:founded(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "org:founded_by(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "org:founded_by(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "org:members(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "org:members(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "org:stateorprovince_of_headquarters(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "org:stateorprovince_of_headquarters(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "org:subsidiaries(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "org:subsidiaries(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "org:top_members/employees(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "org:top_members/employees(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "per:alternate_names(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "per:alternate_names(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "per:cities_of_residence(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "per:cities_of_residence(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "per:countries_of_residence(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "per:countries_of_residence(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "per:country_of_birth(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "per:country_of_birth(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "per:employee_of(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "per:employee_of(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "per:origin(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "per:origin(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "per:spouse(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "per:spouse(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "per:stateorprovinces_of_residence(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "per:stateorprovinces_of_residence(e2,e1)":   #
        mapped_label = ""
    elif kbp37_label == "per:title(e1,e2)":   #
        mapped_label = ""
    elif kbp37_label == "per:title(e2,e1)":   #
        mapped_label = ""

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
        tokens = example["tokens"]

        subj_start, subj_end = get_entity_spans(tokens, example["h"]["name"])
        obj_start, obj_end = get_entity_spans(tokens, example["t"]["name"])

        converted_example = {
            "id": example[0],
            "tokens": tokens,
            "label": example["relation"],
            "grammar": ["SUBJ", "OBJ"],
            "entities": [[subj_start, subj_end], [obj_start, obj_end]]
        }
        if spacy_ner_predictor is not None:
            doc = spacy_ner_predictor(example["tokens"])
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


def get_entity_spans(tokens, entity_text):
    entity_tokens = entity_text.split()
    token_matches = 0
    start = None
    end = None
    for idx, tk in enumerate(tokens):
        if tk == entity_tokens[token_matches]:
            if token_matches == 0:
                start = idx
            token_matches += 1
        else:
            token_matches = 0
        if token_matches == len(entity_tokens):
            end = idx + 1
            break
    if start is not None and end is not None:
        assert entity_text == " ".join(tokens[start:end]), f"{entity_text} vs. {tokens[start:end]}"
        return start, end
    else:
        return None, None


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
        with open(split_path, mode="r", encoding="utf-8") as kbp37_file:

            kbp37_data = list(csv.reader(kbp37_file, delimiter="\t"))[1:]     # skip first header line
            logging.info(f"{len(kbp37_data)} examples in original file")

        logging.info("Processing and exporting to %s", split_export_path)
        converted_examples, num_discarded = kbp37_converter(kbp37_data,
                                                             return_num_discarded=True,
                                                             spacy_ner_predictor=spacy_ner_predictor)

        logging.info(f"{len(converted_examples)} examples in converted file")
        logging.info(f"{num_discarded} examples were discarded during label mapping")

        with open(split_export_path, mode="w", encoding="utf-8") as export_kbp37_file:
            for converted_example in converted_examples:
                export_kbp37_file.write(json.dumps(converted_example))
                export_kbp37_file.write("\n")
        logging.info(utils.get_label_counter(converted_examples))


if __name__ == "__main__":
    main()
