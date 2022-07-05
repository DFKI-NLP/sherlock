import json
import csv
import os
import logging
import argparse

from spacy.attrs import ORTH
from spacy.lang.en import English

import utils
from relation_types import RELATION_TYPES


def map_smiler_label(example):
    smiler_label = example["label"]
    mapped_label = None

    if smiler_label == "birth-place":   # (per, loc)
        mapped_label = "per:place_of_birth"
    # elif smiler_label == "eats":    # (anteaters, ants) so (misc, misc)?
    #     mapped_label = "eats"
    # elif smiler_label == "event-year":    # (event, time), e.g. (Uncial 0301, 5th century)
    #     mapped_label = "event-year"
    # elif smiler_label == "first-product":  # (per/org, misc)
    #     mapped_label = "first-product"
    elif smiler_label == "from-country":    # (per, loc)   perhaps similar "per:country_of_citizenship
        mapped_label = "loc:country_of_origin"  # e.g. (Selden Connor Gile, American)
    elif smiler_label == "has-author":  # (misc, per)
        mapped_label = "per:author"
        example = utils.swap_args(example)
    elif smiler_label == "has-child":   # (per parent, per child)
        mapped_label = "per:children"
    elif smiler_label == "has-edu":  # (per, org)
        mapped_label = "per:schools_attended"
    # elif smiler_label == "has-genre":  # (misc, misc), e.g. (Web browser, Links)
    #     mapped_label = "has-genre"
    # elif smiler_label == "has-height":    # (misc, number)
    #     mapped_label = "has-height"
    # elif smiler_label == "has-highest-mountain":  # (loc, loc mountain)
    #     mapped_label = "has-highest-mountain"
    # elif smiler_label == "has-length":    # (misc, number)
    #     mapped_label = "has-length"
    # elif smiler_label == "has-lifespan":  # (misc, duration)
    #     mapped_label = "has-lifespan"
    elif smiler_label == "has-nationality":  # (per, loc)
        mapped_label = "per:country_of_citizenship"
    elif smiler_label == "has-occupation":  # (per, misc)
        mapped_label = "per:title"
    elif smiler_label == "has-parent":  # (per child, per parent)
        mapped_label = "per:parents"
    # elif smiler_label == "has-population":    # (loc, number)
    #     mapped_label = "has-population"
    elif smiler_label == "has-sibling":     # (per, per)
        mapped_label = "per:siblings"
    elif smiler_label == "has-spouse":   # (per, per)
        mapped_label = "per:spouse"
    # elif smiler_label == "has-tourist-attraction":    # (loc, loc attraction)
    #     mapped_label = "has-tourist-attraction"
    # elif smiler_label == "has-type":  # (misc, type), e.g. <e1>Trice</e1> was a 36 foot <e2>trimaran</e2> sailboat
    #     mapped_label = "has-type"
    # elif smiler_label == "has-weight":    # (misc, number)
    #     mapped_label = "has-weight"
    elif smiler_label == "headquarters":    # (org, loc)
        mapped_label = "org:place_of_headquarters"
    # elif smiler_label == "invented-by":   # (misc, per)
    #     mapped_label = "invented-by"
    # elif smiler_label == "invented-when":  # (misc, time)
    #     mapped_label = "invented-when"
    elif smiler_label == "is-member-of":    # (per, org)
        mapped_label = "org:member_of"
    elif smiler_label == "is-where":    # (org, loc) TODO check NER prefix
        mapped_label = "loc:location_of"
        example = utils.swap_args(example)
    elif smiler_label == "loc-leader":  # (loc, per) TODO check if it fits "per:head_of_gov/state"?
        mapped_label = "per:head_of_gov/state"
    elif smiler_label == "movie-has-director":  # (misc, per)
        mapped_label = "per:director"
        example = utils.swap_args(example)
    elif smiler_label == "no_relation":
        mapped_label = "no_relation"
    elif smiler_label == "org-has-founder":  # (org, per)
        mapped_label = "org:founded_by"
    elif smiler_label == "org-has-member":  # (org, per)
        mapped_label = "org:members"
    elif smiler_label == "org-leader":  # (org, per)
        mapped_label = "org:top_members/employees"
    # elif smiler_label == "post-code":  # (loc, number)
    #     mapped_label = "post-code"
    # elif smiler_label == "starring":  # (misc, per)
    #     mapped_label = "starring"
    # elif smiler_label == "won-award":  # (per, misc)
    #     mapped_label = "won-award"

    if mapped_label is None:
        return None

    assert mapped_label in RELATION_TYPES
    example["label"] = mapped_label
    return example


def smiler_converter(data, word_splitter, return_num_discarded=False):
    num_discarded = 0
    converted_examples = []
    for example in data:
        text = example[4]

        text = text.replace("<e1>", " <e1> ")
        text = text.replace("<e2>", " <e2> ")
        text = text.replace("</e1>", " </e1> ")
        text = text.replace("</e2>", " </e2> ")
        text = text.strip().replace(r"\s\s+", r"\s")
        doc = word_splitter(text)
        tokens = [t.text for t in doc]

        # Handle case where obj may occur before the subj
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

        rel_type = example[3]
        converted_example = map_smiler_label({
            "id": example[0],
            "tokens": tokens,
            "label": rel_type,
            "grammar": ["SUBJ", "OBJ"],
            "entities": [[subj_start, subj_end], [obj_start, obj_end]],
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
        default="../../ds/text/smiler",
        type=str,
        help="path to directory containing the smiler data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/smiler/converted",
        type=str,
        help="path to directory where the converted files should be saved",
    )
    args = parser.parse_args()

    smiler_path = args.data_path
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
    spacy_word_splitter.tokenizer.add_special_case("<e1>", [{ORTH: "<e1>"}])
    spacy_word_splitter.tokenizer.add_special_case("</e1>", [{ORTH: "</e1>"}])
    spacy_word_splitter.tokenizer.add_special_case("<e2>", [{ORTH: "<e2>"}])
    spacy_word_splitter.tokenizer.add_special_case("</e2>", [{ORTH: "</e2>"}])
    for split in ["en-small_corpora_train", "en-small_corpora_test"]:
        split_path = os.path.join(smiler_path, split + ".tsv")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        with open(split_path, mode="r", encoding="utf-8") as smiler_file:

            smiler_data = list(csv.reader(smiler_file, delimiter="\t"))[1:]     # skip first header line
            logging.info(f"{len(smiler_data)} examples in original file")

        logging.info("Processing and exporting to %s", split_export_path)
        converted_examples, num_discarded = smiler_converter(smiler_data, spacy_word_splitter,
                                                             return_num_discarded=True)

        logging.info(f"{len(converted_examples)} examples in converted file")
        logging.info(f"{num_discarded} examples were discarded during label mapping")

        with open(split_export_path, mode="w", encoding="utf-8") as export_smiler_file:
            for converted_example in converted_examples:
                export_smiler_file.write(json.dumps(converted_example))
                export_smiler_file.write("\n")
        logging.info(utils.get_label_counter(converted_examples))


if __name__ == "__main__":
    main()