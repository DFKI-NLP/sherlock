import json
import csv
import os
import logging
import argparse

from spacy.attrs import ORTH
from spacy.lang.en import English

import utils
from relation_types import RELATION_TYPES
from ner_types import NER_TYPES


def map_smiler_label(example, override_entity_types=True):
    smiler_label = example["label"]
    mapped_label = None
    if "type" in example:
        subj_type, obj_type = example["type"]
    else:
        subj_type, obj_type = None, None
    original_types = subj_type, obj_type

    if smiler_label == "birth-place":   # (per, loc)
        mapped_label = "per:place_of_birth"
        subj_type = "PERSON"
        obj_type = "LOC"
    # elif smiler_label == "eats":    # (anteaters, ants) so (misc, misc)?
    #     mapped_label = "eats"
    # elif smiler_label == "event-year":    # (event, time), e.g. (Uncial 0301, 5th century)
    #     mapped_label = "event-year"
    # elif smiler_label == "first-product":  # (per/org, misc)
    #     mapped_label = "first-product"
    elif smiler_label == "from-country":    # (per, loc)   perhaps similar "per:country_of_citizenship
        mapped_label = "per:origin"  # e.g. (Selden Connor Gile, American)
        subj_type = "PERSON"
        obj_type = "LOC"
    elif smiler_label == "has-author":  # (misc, per), only in the big train set, not in the small manually validated
        mapped_label = "per:author"
        example = utils.swap_args(example)
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif smiler_label == "has-child":   # (per parent, per child)
        mapped_label = "per:children"
        subj_type = "PERSON"
        obj_type = "PERSON"
    elif smiler_label == "has-edu":  # (per, org), only in the big train set, not in the small manually validated
        mapped_label = "per:schools_attended"
        subj_type = "PERSON"
        obj_type = "ORG"
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
        subj_type = "PERSON"
        obj_type = "LOC"    # GPE(?)
    elif smiler_label == "has-occupation":  # (per, misc)
        mapped_label = "per:title"
        subj_type = "PERSON"
        obj_type = "POSITION"
    elif smiler_label == "has-parent":  # (per child, per parent)
        mapped_label = "per:parents"
        subj_type = "PERSON"
        obj_type = "PERSON"
    # elif smiler_label == "has-population":    # (loc, number)
    #     mapped_label = "has-population"
    elif smiler_label == "has-sibling":     # (per, per)
        mapped_label = "per:siblings"
        subj_type = "PERSON"
        obj_type = "PERSON"
    elif smiler_label == "has-spouse":   # (per, per)
        mapped_label = "per:spouse"
        subj_type = "PERSON"
        obj_type = "PERSON"
    # elif smiler_label == "has-tourist-attraction":    # (loc, loc attraction)
    #     mapped_label = "has-tourist-attraction"
    # elif smiler_label == "has-type":  # (misc, type), e.g. <e1>Trice</e1> was a 36 foot <e2>trimaran</e2> sailboat
    #     mapped_label = "has-type"
    # elif smiler_label == "has-weight":    # (misc, number)
    #     mapped_label = "has-weight"
    elif smiler_label == "headquarters":    # (org, loc)
        mapped_label = "org:place_of_headquarters"
        subj_type = "ORG"
        obj_type = "LOCATION"
    # elif smiler_label == "invented-by":   # (misc, per)
    #     mapped_label = "invented-by"
    # elif smiler_label == "invented-when":  # (misc, time)
    #     mapped_label = "invented-when"
    elif smiler_label == "is-member-of":    # (per, org)
        mapped_label = "org:member_of"
        example = utils.swap_args(example)
        subj_type = "ORG"
        obj_type = "PERSON"
    elif smiler_label == "is-where":    # (org, loc)
        mapped_label = "loc:location_of"
        example = utils.swap_args(example)
        subj_type = "LOC"
        obj_type = "ORG"
    elif smiler_label == "loc-leader":  # (loc, per), only in the big train set, not in the small manually validated
        mapped_label = "per:head_of_gov/state"  # TODO check
        example = utils.swap_args(example)
        subj_type = "PERSON"
        obj_type = "LOC"
    elif smiler_label == "movie-has-director":  # (misc, per)
        mapped_label = "per:director"
        example = utils.swap_args(example)
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif smiler_label == "no_relation":
        mapped_label = "no_relation"
    elif smiler_label == "org-has-founder":  # (org, per)
        mapped_label = "org:founded_by"
        subj_type = "ORG"
        obj_type = "PERSON"
    elif smiler_label == "org-has-member":  # (org, per)
        mapped_label = "org:members"
        subj_type = "ORG"
        obj_type = "PERSON"
    elif smiler_label == "org-leader":  # (org, per)
        mapped_label = "org:top_members/employees"
        subj_type = "ORG"
        obj_type = "PERSON"
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
    if "type" in example:
        if not override_entity_types:
            subj_type, obj_type = original_types
        example["type"] = [map_smiler_ner_label(subj_type), map_smiler_ner_label(obj_type)]
    return example


def map_smiler_ner_label(smiler_label):
    # not really necessary since we either do not include NER labels or use correct NER labels from plass ner model
    mapped_label = smiler_label
    # if smiler_label == "PER":
    #     mapped_label = "PERSON"
    # elif smiler_label == "ORG":
    #     mapped_label = "ORG"

    assert mapped_label in NER_TYPES, f"{mapped_label} not valid label"
    return mapped_label


def smiler_converter(data, word_splitter, return_num_discarded=False, spacy_ner_predictor=None):
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
        converted_example = {
            "id": example[0],
            "tokens": tokens,
            "label": rel_type,
            "grammar": ["SUBJ", "OBJ"],
            "entities": [[subj_start, subj_end], [obj_start, obj_end]]
        }
        converted_examples.append(converted_example)
    converted_examples = utils.predict_entity_type(spacy_ner_predictor=spacy_ner_predictor,
                                                   examples=converted_examples)
    final_examples = []
    for converted_example in converted_examples:
        converted_example = map_smiler_label(converted_example)
        if converted_example is not None:
            final_examples.append(converted_example)
        else:
            num_discarded += 1
    if return_num_discarded:
        return final_examples, num_discarded
    else:
        return final_examples


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
    parser.add_argument(
        "--ner_model_path",
        # default="./models/spacy_trf/model-best",
        type=str,
        help="path to ner model",
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

    spacy_ner_predictor = utils.load_spacy_predictor(args.ner_model_path) if args.ner_model_path else None
    # possible TODO is to combine the spacy ner predictor and the word splitter
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

        with open(split_export_path, mode="w", encoding="utf-8") as export_smiler_file:
            for converted_example in converted_examples:
                export_smiler_file.write(json.dumps(converted_example))
                export_smiler_file.write("\n")
        logging.info(utils.get_label_counter(converted_examples))


if __name__ == "__main__":
    main()
