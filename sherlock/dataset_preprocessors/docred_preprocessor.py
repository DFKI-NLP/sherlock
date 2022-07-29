import json
import os
import logging
import argparse

import utils
from relation_types import RELATION_TYPES
from ner_types import NER_TYPES
from relation_ner_mapping import get_entity_types_from_relation


def map_docred_label(example, override_entity_types=False, adjust_entity_type=True):
    docred_label = example["label"]
    mapped_label = None
    if "type" in example:
        subj_type, obj_type = map_docred_ner_label(example["type"][0]), map_docred_ner_label(example["type"][1])
    else:
        subj_type, obj_type = None, None

    if docred_label in [
        "after a work by",  # (misc, per)   fewrel
        "applies to jurisdiction",  # (org, loc)
        "architect",    # (misc, per)   fewrel
        "award received",   # (per, misc)
        "basin country",    # (loc, loc)
        "cast member",  # (misc, per)
        "characters",   # (misc, per)
        "contains administrative territorial entity",   # (loc, contained loc)
        "continent",    # (loc, continent)
        "distributor",  # (misc, distributor org) fewrel
        "end time",  # (misc, time)
        "followed by",  # (org, successor)
        "follows",  # (successor, org)
        "genre",    # (org, org genre)
        "has part",  # (org, per)
        "heritage designation",    # (misc, designation) Jaeckel Hotel added to National Register of Historic Places
        "influenced by",    # (misc, per)
        "instance of",  # (org, org instance)
        "instrument",  # (per, misc) fewrel
        "language of work or name",  # (misc, misc language)
        "languages spoken, written or signed",  # (per, misc language)
        "league",   # (org, misc league)
        "legislative body",  # (loc, org)
        "licensed to broadcast to",  # (wpgg, atlantic city) WSJO added the programming of WPGG 1450 Atlantic City on...
        "located in or next to body of water",  # (loc, loc body of water)
        "member of sports team",    # (per, org)
        "military branch",  # (per, org branch)
        "mountain range",   # (loc mount korbu, loc ...) Mount Korbu , the tallest mountain of the Titiwangsa Mountains
        "mouth of the watercourse",  # (loc, loc), <head>Second River joins the <tail>Passaic River
        "movement",  # (per, misc) fewrel
        "narrative location",   # (misc, loc)
        "nominated for",    # (misc a star is born, misc best picture)
        "occupant",  # (loc silverstein eye centers arena, org kansas city mavericks)
        "official language",    # (loc, org language)
        "operating system",  # (misc, misc operating system)
        "operator",     # (loc, org operator)
        "original language of work",    # (misc, loc language)
        "original network",  # (misc, org network)
        "parent taxon",  # (misc, parent misc)
        "part of",  # (part loc, loc)
        "participant",  # (misc, per participant)
        "participant of",   # (per participant, misc)
        "participating team",   # (org Bundesliga, org Werder Bremen)
        "place served by transport hub",    # (transport, loc)
        "platform",  # (misc, misc platform)
        "point in time",    # (time point, time)
        "present in work",  # (per, misc)
        "publication date",  # (misc, time)
        "record label",  # (org, org record label)
        "replaced by",  # (loc, loc replacement)
        "replaces",  # (loc replacement, loc)
        "said to be the same as",  # (misc, misc) fewrel
        "separated from",   # (org separated from, org) Christianity separated from Judaism
        "series",   # (misc, misc series)
        "sports season of league or competition",   # (time, org)
        "start time",    # (misc, time)
        "subclass of",  # (misc subclass, misc)
        "successful candidate",  # (misc, per)  fewrel
        "taxon rank",   # (misc muscidae, misc family) from the fly family Muscidae
        "territory claimed by",  # (loc Taiwan, loc China)
        "tributary",    # (loc hudson river, loc schroon river) fewrel
        "winner",   # (misc, per)
        "work location",    # (per, loc) geographical location, e.g. Gianni Alemanno, Mayor of Rome
    ]:
        return None
    elif docred_label == "author":     # (misc, per)
        mapped_label = "per:author"
        example = utils.swap_args(example)
        obj_type = "WORK_OF_ART" if adjust_entity_type else obj_type
    elif docred_label == "capital of":    # (capital, _)
        mapped_label = "loc:capital_of"
    elif docred_label == "capital":    # (_, capital)
        mapped_label = "loc:capital_of"
        example = utils.swap_args(example)
    elif docred_label == "chairperson":    # (org, per)
        mapped_label = "org:top_members/employees"
    elif docred_label == "child":    # (parent, child)
        mapped_label = "per:children"
    elif docred_label == "composer":   # (misc, per)
        mapped_label = "per:composer"
        example = utils.swap_args(example)
        obj_type = "WORK_OF_ART" if adjust_entity_type else obj_type
    elif docred_label == "conflict":   # (per/org, misc conflict)
        mapped_label = "event:conflict"
        example = utils.swap_args(example)
        subj_type = "EVENT" if adjust_entity_type else subj_type
    elif docred_label == "country":    # (org/loc, loc country of)
        mapped_label = "loc:country"
    elif docred_label == "country of citizenship":     # (per, loc)
        mapped_label = "per:country_of_citizenship"
    elif docred_label == "country of origin":  # (misc/org/per, loc)
        mapped_label = "loc:country_of_origin"
        example = utils.swap_args(example)
    elif docred_label == "creator":    # (misc, per)
        mapped_label = "per:creator"
        example = utils.swap_args(example)
        obj_type = "WORK_OF_ART" if adjust_entity_type else obj_type
    elif docred_label == "date of birth":    # (per, time)
        mapped_label = "per:date_of_birth"
    elif docred_label == "date of death":    # (per, time)
        mapped_label = "per:date_of_death"
    elif docred_label == "developer":  # (misc, per/org)
        if obj_type == "ORG":
            mapped_label = "org:developer"
            example = utils.swap_args(example)
    elif docred_label == "director":   # (misc, per)
        mapped_label = "per:director"
        example = utils.swap_args(example)
        obj_type = "WORK_OF_ART" if adjust_entity_type else obj_type
    elif docred_label == "dissolved, abolished or demolished":   # (org, time)
        mapped_label = "org:dissolved"
    elif docred_label == "educated at":   # (per, org)
        mapped_label = "per:schools_attended"
    elif docred_label == "employer":  # (per, org)
        mapped_label = "per:employee_of"
    # elif docred_label == "ethnic group":   # (per/loc, loc ethnic group) or (per, NORP?) TODO check NER
    #     # most of the examples seem to be something like (loc Australia, loc Australian)
    #     mapped_label = "per:ethnic_group"
    elif docred_label in ["father", "mother"]:     # (child, father/mother)
        mapped_label = "per:parents"
    elif docred_label == "field of work":  # (per, misc) "German <misc>botanist <per>Conrad Moench"
        # some examples are compatible with position/title
        # but there are also examples such as (Robert Robinson, Organic Chemistry)
        mapped_label = "per:field_of_work"
    elif docred_label == "founded by":     # (org, per)
        mapped_label = "org:founded_by"
    elif docred_label in ["head of government", "head of state"]:  # head of gov (loc, per), head of state
        mapped_label = "per:head_of_gov/state"
        example = utils.swap_args(example)
    elif docred_label == "headquarters location":   # (org, loc)
        mapped_label = "org:place_of_headquarters"
    elif docred_label == "inception":    # (loc/org/misc, time)
        if subj_type == "ORG":
            mapped_label = "org:founded"
    elif docred_label == "language":
        mapped_label = "per:language"
    elif docred_label in [
        "located in the administrative territorial entity",   # (loc, a.t.e) Memphis , Scotland County
        "located on terrain feature",   # (loc, terrain feature) Kanatadika on Euboea
        # "located in or next to body of water"     # (loc, body of water)
    ]:
        mapped_label = "loc:located_in"
    elif docred_label == "location of formation":  # (org, loc)
        mapped_label = "org:location_of_formation"
    elif docred_label == "lyrics by":  # (song, per writer)
        mapped_label = "per:lyrics_by"
        example = utils.swap_args(example)
        obj_type = "WORK_OF_ART" if adjust_entity_type else obj_type
    # elif doc_red_label == "location":  # (misc, loc) example: Second World War in Europe
    #     mapped_label = "location"  # TODO name, (fac/event/item, loc) need NER to determine NER prefix for RE label
    elif docred_label == "manufacturer":  # (misc, manufacturer org)
        mapped_label = "org:product_or_technology_or_service"
        example = utils.swap_args(example)
    elif docred_label == "member of":    # (per/org/loc, org)
        if subj_type in ["PER", "PERSON"]:
            mapped_label = "per:member_of"
        else:
            mapped_label = "org:members"
            example = utils.swap_args(example)
    elif docred_label == "member of political party":  # (per, org)
        mapped_label = "per:political_affiliation"
    elif docred_label == "notable work":   # (per, misc)
        mapped_label = "per:notable_work"
        obj_type = "WORK_OF_ART" if adjust_entity_type else obj_type
    elif docred_label == "occupation":  # (per, misc)
        mapped_label = "per:title"
    elif docred_label == "owned by":   # (org, per/org owner)
        mapped_label = "org:shareholders"
    elif docred_label == "parent organization":    # (parent org, org)
        mapped_label = "org:parents"    # (daughter company, parent company)
        example = utils.swap_args(example)
    elif docred_label == "performer":  # (misc, org/per performer)
        mapped_label = "per:performer"
        example = utils.swap_args(example)
        obj_type = "WORK_OF_ART" if adjust_entity_type else obj_type
    elif docred_label == "place of birth":  # (per, loc)
        mapped_label = "per:place_of_birth"
    elif docred_label == "place of death":  # (per, loc)
        mapped_label = "per:place_of_death"
    elif docred_label == "position held":  # (per, misc)
        mapped_label = "per:title"
    elif docred_label == "producer":   # (misc, per) Bad produced by Quincy Jones
        mapped_label = "per:producer"
    elif docred_label == "product or material produced":  # (org, misc)
        mapped_label = "org:product_or_technology_or_service"
    elif docred_label == "production company":  # (misc, org)  Atomic Blonde produced by Focus Features
        mapped_label = "org:production_company"
    # elif doc_red_label == "publisher":    # (misc, org/per)
    #     mapped_label = "publisher"  # TODO name, need NER to determine NER prefix for RE label
    elif docred_label == "religion":   # (per, org religion)
        mapped_label = "per:religion"
    elif docred_label == "residence":  # (per, loc)
        mapped_label = "per:places_of_residence"
    elif docred_label == "screenwriter":   # (misc, per)
        mapped_label = "per:screenwriter"
        example = utils.swap_args(example)
        obj_type = "WORK_OF_ART" if adjust_entity_type else obj_type
    elif docred_label == "sibling":    # (per, per)
        mapped_label = "per:siblings"
    elif docred_label == "sister city":    # (loc, loc)
        mapped_label = "loc:twinned_adm_body"
    elif docred_label == "spouse":    # (per, per)
        mapped_label = "per:spouse"
    elif docred_label == "subsidiary":  # (org parent, org subsidiary)
        mapped_label = "org:subsidiaries"
    elif docred_label == "unemployment rate":  # (loc, num)
        mapped_label = "loc:unemployment_rate"
    elif docred_label == "work location":  # (per, loc)
        mapped_label = "per:work_location"

    if mapped_label is None:
        return None

    assert mapped_label in RELATION_TYPES, f"mapped_label='{mapped_label}' not in RELATION_TYPES"
    example["label"] = mapped_label
    if override_entity_types:
        subj_type, obj_type = get_entity_types_from_relation(mapped_label, subj_type, obj_type)
        example["type"] = [map_docred_ner_label(subj_type), map_docred_ner_label(obj_type)]
    if subj_type is not None and obj_type is not None:
        example["type"] = [map_docred_ner_label(subj_type), map_docred_ner_label(obj_type)]
    return example


def map_docred_ner_label(docred_label):
    mapped_label = docred_label
    if docred_label == "PER":
        mapped_label = "PERSON"
    elif docred_label == "ORG":
        mapped_label = "ORG"
    elif docred_label == "LOC":
        mapped_label = "LOC"
    elif docred_label == "MISC":
        mapped_label = "MISC"
    elif docred_label == "TIME":
        mapped_label = "TIME"
    elif docred_label == "NUM":
        mapped_label = "CARDINAL"   # TODO NUM is not in plass ner label set

    if mapped_label is not None:
        assert mapped_label in NER_TYPES, f"{mapped_label} not valid label"
    return mapped_label


def docred_converter(example, docred_rel_info, return_num_discarded=False):
    num_discarded = 0
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
        # TODO not sure how to count original number of examples for document wide re annotation for statistics
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
            converted_example = map_docred_label({
                "id": "r/" + utils.generate_example_id(),
                "tokens": example["sents"][sent_id],
                "label": rel_type,
                "grammar": ["SUBJ", "OBJ"],
                "entities": [[subj_start, subj_end], [obj_start, obj_end]],
                "type": [subj_type, obj_type]
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

        num_discarded = 0
        converted_examples = []
        for example in docred_data:
            if "labels" in example:
                converted_exs, num_discrd = docred_converter(example, docred_rel_info, return_num_discarded=True)
                converted_examples += converted_exs
                num_discarded += num_discrd
        logging.info(f"{len(converted_examples)} examples in converted file")
        logging.info(f"{num_discarded} examples were discarded during label mapping")

        with open(split_export_path, mode="w", encoding="utf-8") as f:
            for conv_example in converted_examples:
                f.write(json.dumps(conv_example))
                f.write("\n")
        logging.info(utils.get_label_counter(converted_examples))


if __name__ == "__main__":
    main()
