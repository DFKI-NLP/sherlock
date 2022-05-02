import json
import csv
import os
import logging
import argparse

from spacy.lang.en import English

import utils


def replace_underscore_in_span(text, start, end):
    cleaned_text = text[:start] + text[start:end].replace("_", " ") + text[end:]
    return cleaned_text


def gids_converter(example, word_splitter, replace_underscores=True):
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
    return {
        "id": "r/" + utils.generate_example_id(),
        "tokens": word_tokens,
        "label": rel_type,
        "grammar": ["SUBJ", "OBJ"],
        "entities": [[subj_span.start, subj_span.end], [obj_span.start, obj_span.end]],
        # "type": [subj_type, obj_type]
    }


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
        logging.info("Processing and exporting to %s", split_export_path)
        with open(split_path, mode="r", encoding="utf-8") as gids_file, \
                open(split_export_path, mode="w", encoding="utf-8") as export_gids_file:
            gids_data = csv.reader(gids_file, delimiter="\t")
            for example in gids_data:
                converted_example = gids_converter(example, spacy_word_splitter, replace_underscores=True)
                export_gids_file.write(json.dumps(converted_example))
                export_gids_file.write("\n")


if __name__ == "__main__":
    main()
