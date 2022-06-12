import json
import csv
import os
import logging
import argparse

from spacy.lang.en import English

import utils
from relation_types import RELATION_TYPES


def map_gids_label(example):
    gids_label = example["label"]
    mapped_label = None

    if gids_label == "/people/person/education./education/education/degree":
        mapped_label = "per:degree"
    elif gids_label == "NA":
        mapped_label = "no_relation"
    # TODO check whether to include the following labels as well
    elif gids_label == "/people/person/education./education/education/institution":
        mapped_label = "per:schools_attended"
    elif gids_label == "/people/person/place_of_birth":
        mapped_label = "per:place_of_birth"
    elif gids_label == "/people/deceased_person/place_of_death":
        mapped_label = "per:place_of_death"

    if mapped_label is None:
        return None

    assert mapped_label in RELATION_TYPES
    example["label"] = mapped_label
    return example


def replace_underscore_in_span(text, start, end):
    cleaned_text = text[:start] + text[start:end].replace("_", " ") + text[end:]
    return cleaned_text


def gids_converter(data, word_splitter, replace_underscores=True, return_num_discarded=False):
    num_discarded = 0
    converted_examples = []
    for example in data:
        text = example[5].strip()[:-9].strip()  # remove '###END###' from text
        subj_text = example[2]
        obj_text = example[3]
        subj_char_start = text.find(subj_text)
        assert subj_char_start != -1, f"Did not find <{subj_text}> in the text"
        subj_char_end = subj_char_start + len(subj_text)
        obj_char_start = text.find(obj_text)
        assert obj_char_start != -1, f"Did not find <{obj_text}> in the text"
        obj_char_end = obj_char_start + len(obj_text)
        if replace_underscores:
            text = replace_underscore_in_span(text, subj_char_start, subj_char_end)
            text = replace_underscore_in_span(text, obj_char_start, obj_char_end)
        doc = word_splitter(text)
        word_tokens = [t.text for t in doc]
        subj_span = doc.char_span(subj_char_start, subj_char_end, alignment_mode="expand")
        obj_span = doc.char_span(obj_char_start, obj_char_end, alignment_mode="expand")

        rel_type = example[4]
        converted_example = map_gids_label({
            "id": "r/" + utils.generate_example_id(),
            "tokens": word_tokens,
            "label": rel_type,
            "grammar": ["SUBJ", "OBJ"],
            "entities": [[subj_span.start, subj_span.end], [obj_span.start, obj_span.end]],
            # "type": [subj_type, obj_type]
        })
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
        default="../../ds/text/gids",
        type=str,
        help="path to directory containing the gids data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/gids/converted",
        type=str,
        help="path to directory where the converted files should be saved",
    )
    args = parser.parse_args()

    gids_path = args.data_path
    export_path = args.export_path
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    spacy_word_splitter = English()
    for split in ["train", "dev", "test"]:
        split_path = os.path.join(gids_path, split + ".tsv")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        with open(split_path, mode="r", encoding="utf-8") as gids_file:

            gids_data = list(csv.reader(gids_file, delimiter="\t"))
            logging.info(f"{len(gids_data)} examples in original file")

        logging.info("Processing and exporting to %s", split_export_path)
        converted_examples, num_discarded = gids_converter(gids_data, spacy_word_splitter, replace_underscores=True,
                                                           return_num_discarded=True)

        logging.info(f"{len(converted_examples)} examples in converted file")
        logging.info(f"{num_discarded} examples were discarded during label mapping")

        with open(split_export_path, mode="w", encoding="utf-8") as export_gids_file:
            for converted_example in converted_examples:
                export_gids_file.write(json.dumps(converted_example))
                export_gids_file.write("\n")
        logging.info(utils.get_label_counter(converted_examples))


if __name__ == "__main__":
    main()
