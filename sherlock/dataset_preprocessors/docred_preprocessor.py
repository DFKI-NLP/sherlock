import json
import os
import logging
import argparse

import utils
from relation_types import RELATION_TYPES


def map_doc_red_label(example):
    doc_red_label = example["label"]
    mapped_label = None

    if doc_red_label in [
        "applies to jurisdiction",  # (org, loc)
        "award received",   # (per, misc)
        "basin country",    # (loc, loc)
        "cast_member",  # (misc, per)
        "chairperson",  # (org, per)    TODO could be relevant
        "characters",   # (misc, per)
        "child",    # (parent, child)   TODO
        "contains administrative territorial entity",   # (loc, contained loc)
        "continent",    # (loc, continent)
        "country",  # (org, loc)
        "date_of_birth",    # (per, time)   TODO
        "date_of_death",  # (per, time)   TODO
        "dissolved, abolished or demolished",   # (org, time)
        "educated at",  # (per, org)    TODO
        "employer",  # (per, org)   TODO
        "end time",  # (misc, time)
        "father",   # (child, father)   TODO
        "followed by",  # (org, successor)
        "follows",  # (successor, org)
        "founded by",   # (org, per)    TODO
        "genre",    # (org, org genre)
        "has part",  # (org, per)
        "headquarters location",    # (org, loc) TODO
        "inception",    # (loc, time)
        "influenced by",    # (misc, per)
        "instance of",  # (org, org instance)
        "languages spoken, written or signed",  # (per, misc)
        "league",   # (org, misc league)
        "legislative body",  # (loc, org)
        "located in or next to body of water",  # (loc, loc body of water)
        "location",  # (misc, loc) example: Second World War in Europe TODO
        "member of",    # (per, org)
        "member of political party",    # (per, org) TODO religious/political affiliation?
        "member of sports team",    # (per, org)
        "military branch",  # (per, org branch)
        "mother",   # (child, mother)   TODO
        "mouth of the watercourse",  # (loc, loc), <head>Second River joins the <tail>Passaic River
        "narrative location",   # (misc, loc)
        "official language",    # (loc, org language)
        "operator",     # (loc, org operator)
        "original language of work",    # (misc, loc language)
        "original network",  # (misc, org network)
        "owned by",  # (org, per owner) TODO
        "parent organization",  # (parent org, org) TODO
        "parent taxon",  # (misc, parent misc)
        "part of",  # (part loc, loc)
        "participant",  # (misc, per participant)
        "participant of",   # (per participant, misc)
        "place of birth",   # (per, loc) TODO
        "place of death",   # (per, loc) TODO
        "platform",  # (misc, misc platform)
        "point in time",    # (time point, time)
        "present in work",  # (per, misc)
        "publication date",  # (misc, time)
        "record label",  # (org, org record label)
        "replaced by",  # (loc, loc replacement)
        "replaces",  # (loc replacement, loc)
        "residence",    # (per, loc) TODO
        "separated_from",   # (org separated from, org) Christianity separated from Judaism
        "series",   # (misc, misc series)
        "sibling",  # (per, per)    TODO
        "spouse",   # (per, per)    TODO
        "start time",    # (misc, time)
        "subclass of",  # (misc subclass, misc)
        "subsidiary",   # (org, org subsidiary) TODO
        "territory claimed by",  # (loc Taiwan, loc China)
    ]:
        return None
    elif doc_red_label == "author":     # (misc, per)
        mapped_label = "per:author"
    elif doc_red_label == "capital of":    # (capital, _)
        mapped_label = "loc:capital_of"
    elif doc_red_label == "capital":    # (_, capital)
        mapped_label = "loc:capital_of"
    elif doc_red_label == "composer":   # (misc, per)
        mapped_label = "per:composer"
        example = utils.swap_args(example)
    elif doc_red_label == "conflict":   # (per, misc)   TODO check
        mapped_label = "per:conflict"
    elif doc_red_label == "contains location":
        mapped_label = "loc:contains_location"
    elif doc_red_label == "country of citizenship":     # (per, loc)
        mapped_label = "per:country_of_citizenship"
    elif doc_red_label == "country of origin":  # (misc, loc)
        mapped_label = "loc:country_of_origin"
        example = utils.swap_args(example)
    elif doc_red_label == "creator":    # (misc, per)
        mapped_label = "per:creator"
        example = utils.swap_args(example)
    elif doc_red_label == "developer":  # (misc, per)
        mapped_label = "per:developer"
        example = utils.swap_args(example)
    elif doc_red_label == "director":   # (misc, per)
        mapped_label = "per:director"
        example = utils.swap_args(example)
    elif doc_red_label == "ethnic group":   # (loc, loc ethnic group) or (per, NORP?) TODO check
        mapped_label = "per:ethnic_group"
    elif doc_red_label in ["head of government", "head of state"]:  # head of gov (loc, per), head of state
        mapped_label = "per:head_of_gov/state"
        example = utils.swap_args(example)
    elif doc_red_label == "language":
        mapped_label = "per:language"
    elif doc_red_label in [
            "located in the administrative territorial entity",     # (loc, administrative territorial entity)
            "located on terrain feature",   # (loc, terrain feature)
            # "located in or next to body of water"     # (loc, body of water)
    ]:
        mapped_label = "org:facility_or_location"   # TODO not a really good fit
    elif doc_red_label == "location of formation":  # (org, loc)
        mapped_label = "org:location_of_formation"
    elif doc_red_label == "lyrics by":  # (song, per writer) TODO is this really relevant?
        mapped_label = "per:lyrics_by"
        example = utils.swap_args(example)
    # elif doc_red_label == "location":
    #     mapped_label = "location"  # TODO name, (fac/event/item, loc) need NER to determine NER prefix for RE label
    elif doc_red_label == "manufacturer":  # (misc, manufacturer org)
        mapped_label = "org:product_or_technology_or_service"
        example = utils.swap_args(example)
    elif doc_red_label == "notable work":   # (per, misc)
        mapped_label = "per:notable_work"
    elif doc_red_label == "performer":  # (misc, org/per performer)
        mapped_label = "per:performer"
    elif doc_red_label == "position held":  # (per, misc)
        mapped_label = "per:title"
    elif doc_red_label == "producer":   # (misc, per) Bad produced by Quincy Jones
        mapped_label = "per:producer"
    elif doc_red_label == "product or material produced":  # (org, misc)
        mapped_label = "org:product_or_technology_or_service"
    elif doc_red_label == "production company":  # (misc, org)  Atomic Blonde produced by Focus Features
        mapped_label = "org:production_company"
    # elif doc_red_label == "publisher":    # (misc, org/per)
    #     mapped_label = "publisher"  # TODO name, need NER to determine NER prefix for RE label
    elif doc_red_label == "religion":   # (per, org religion)
        mapped_label = "per:religion"
    elif doc_red_label == "screenwriter":   # (misc, per)
        mapped_label = "per:screenwriter"
        example = utils.swap_args(example)
    elif doc_red_label == "sister city":    # (loc, loc)
        mapped_label = "loc:twinned_adm_body"
    elif doc_red_label == "unemployment rate":  # (loc, num)
        mapped_label = "loc:unemployment_rate"
    elif doc_red_label == "work location":  # (per, loc)
        mapped_label = "per:work_location"

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
