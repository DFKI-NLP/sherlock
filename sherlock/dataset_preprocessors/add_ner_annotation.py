import logging
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import utils
from ner_types import NER_TYPES
from relation_types import RELATION_TYPES

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_entity_types_from_relation(relation_label, subj_type=None, obj_type=None):
    assert relation_label in RELATION_TYPES
    if relation_label == "per:place_of_birth":
        subj_type = "PERSON"
        obj_type = "LOC"
    elif relation_label == "per:place_of_death":
        subj_type = "PERSON"
        obj_type = "LOC"
    elif relation_label == "org:alternate_names":
        subj_type = "ORG"
        obj_type = "ORG"
    elif relation_label == "org:founded":
        subj_type = "ORG"
        obj_type = "DATE"
    elif relation_label == "org:founded_by":
        subj_type = "ORG"
        obj_type = "PERSON"
    elif relation_label == "org:members":
        subj_type = "ORG"
        obj_type = "ORG"  # may be a country -> LOC/GPE
    elif relation_label == "org:subsidiaries":
        subj_type = "ORG"
        obj_type = "ORG"
    elif relation_label == "org:top_members/employees":
        subj_type = "ORG"
        obj_type = "PERSON"
    elif relation_label == "per:alternate_names":
        subj_type = "PERSON"
        obj_type = "PERSON"
    elif relation_label == "per:places_of_residence":
        subj_type = "PERSON"
        obj_type = "LOC"
    elif relation_label == "per:date_of_birth":
        subj_type = "PERSON"
        obj_type = "DATE"
    elif relation_label == "per:date_of_death":
        subj_type = "PERSON"
        obj_type = "DATE"
    elif relation_label == "per:employee_of":
        subj_type = "PERSON"
        obj_type = "ORG"
    elif relation_label == "per:origin":
        subj_type = "PERSON"
        obj_type = "LOC"
    elif relation_label == "per:political_affiliation":
        subj_type = "PERSON"
        obj_type = "ORG"
    elif relation_label == "per:title":
        subj_type = "PERSON"
        obj_type = "POSITION"
    elif relation_label == "per:author":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "per:children":
        subj_type = "PERSON"
        obj_type = "PERSON"
    elif relation_label == "per:schools_attended":
        subj_type = "PERSON"
        obj_type = "ORG"
    elif relation_label == "per:country_of_citizenship":
        subj_type = "PERSON"
        obj_type = "LOC"  # GPE(?)
    elif relation_label == "per:parents":
        subj_type = "PERSON"
        obj_type = "PERSON"
    elif relation_label == "per:siblings":
        subj_type = "PERSON"
        obj_type = "PERSON"
    elif relation_label == "per:spouse":
        subj_type = "PERSON"
        obj_type = "PERSON"
    elif relation_label == "org:place_of_headquarters":
        subj_type = "ORG"
        obj_type = "LOC"
    elif relation_label == "org:member_of":
        subj_type = "ORG"
        obj_type = "ORG"
    elif relation_label == "per:member_of":
        subj_type = "PERSON"
        obj_type = "ORG"
    elif relation_label == "loc:location_of":
        subj_type = "LOC"
        obj_type = "ORG"
        if obj_type is not None and obj_type not in ["PERSON", "ORG", "MISC", "WORK_OF_ART"]:
            obj_type = "PERSON"  # TODO ambivalent
    elif relation_label == "per:head_of_gov/state":
        subj_type = "PERSON"
        obj_type = "LOC"
    elif relation_label == "per:director":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "org:members":
        subj_type = "ORG"
        obj_type = "ORG"
    elif relation_label == "org:top_members/employees":
        subj_type = "ORG"
        obj_type = "PERSON"
    elif relation_label == "loc:capital_of":
        subj_type = "LOC"
        obj_type = "LOC"  # could be GPE
    elif relation_label == "per:composer":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "event:conflict":
        subj_type = "EVENT"
        if obj_type != "ORG":
            obj_type = "PERSON"
    elif relation_label == "loc:country":
        subj_type = "LOC"  # TODO org/loc
        obj_type = "LOC"
    elif relation_label == "loc:country_of_origin":
        subj_type = "LOC"
        if obj_type is not None and obj_type not in ["MISC", "ORG", "PERSON"]:
            obj_type = "MISC"  # TODO misc/org/per
    elif relation_label == "per:creator":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "per:developer":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "org:dissolved":
        subj_type = "ORG"
        obj_type = "DATE"
    elif relation_label == "per:ethnic_group":
        if subj_type not in ["LOC", "PERSON"]:
            subj_type = "PERSON"
        obj_type = "LOC"
    elif relation_label == "per:field_of_work":
        subj_type = "PERSON"
        obj_type = "MISC"
    elif relation_label == "per:language":
        subj_type = "PERSON"
        obj_type = "MISC"
    elif relation_label == "org:facility_or_location":
        subj_type = "ORG"
        obj_type = "LOC"
    elif relation_label == "org:location_of_formation":
        subj_type = "ORG"
        obj_type = "LOC"
    elif relation_label == "per:lyrics_by":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "org:product_or_technology_or_service":
        subj_type = "ORG"
        obj_type = "PRODUCT"
    elif relation_label == "per:notable_work":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "org:shareholders":
        subj_type = "ORG"
        if obj_type not in ["PERSON", "ORGANIZATION"]:
            obj_type = "PERSON"
    elif relation_label == "org:parents":
        subj_type = "ORG"
        obj_type = "ORG"
    elif relation_label == "per:performer":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "per:producer":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "org:production_company":
        subj_type = "ORG"
        obj_type = "WORK_OF_ART"
    elif relation_label == "per:religion":
        subj_type = "PERSON"
        obj_type = "ORG"
    elif relation_label == "per:screenwriter":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "loc:twinned_adm_body":
        subj_type = "LOC"
        obj_type = "LOC"
    elif relation_label == "loc:unemployment_rate":
        subj_type = "LOC"
        obj_type = "NUM"
    elif relation_label == "per:work_location":
        subj_type = "PERSON"
        obj_type = "LOC"
    elif relation_label == "loc:located_in":
        subj_type = "LOC"
        obj_type = "LOC"

    if subj_type is not None and obj_type is not None:
        assert subj_type in NER_TYPES and obj_type in NER_TYPES
    else:
        logging.debug(f"Did not map from [{relation_label}] to any NER labels for subject and object")
    return subj_type, obj_type


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    remove_o_tags = False
    override_entity_types = False

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/collated/train.jsonl",
        type=str,
        help="path to data file",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/collated/ner-annotated/train.jsonl",
        type=str,
        help="path to export file",
    )
    parser.add_argument(
        "--spacy_batch_size",
        default=100,
        type=int,
        help="batch size for spaCy prediction",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="batch size for processing (for progress report)",
    )
    parser.add_argument(
        "--ner_model_path",
        default="../../models/spacy_trf/model-best",
        type=str,
        help="path to ner model",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    export_path = Path(args.export_path)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    batch_size = args.batch_size
    spacy_batch_size = args.spacy_batch_size
    spacy_ner_predictor = utils.load_spacy_predictor(args.ner_model_path)

    examples = []
    annotated_examples = []
    with open(data_path, mode="r", encoding="utf8") as in_file:
        for line in in_file.readlines():
            example = json.loads(line)
            examples.append(example)
        i = 0
        # batch processing
        for j in tqdm(range(int(len(examples) / batch_size))):
            logging.debug(f"Processing batch {j}/{int(len(examples) / batch_size)}")
            batch = utils.predict_entity_type(spacy_ner_predictor, examples[i:i+batch_size],
                                              batch_size=spacy_batch_size)
            # remove entity type field if O is predicted for any of the head or tail entity
            if remove_o_tags or override_entity_types:
                for example in batch:
                    if override_entity_types:
                        subj_type, obj_type = get_entity_types_from_relation(relation_label=example["relation"],
                                                                             subj_type=example["type"][0],
                                                                             obj_type=example["type"][1])
                        example["type"] = [subj_type, obj_type]
                    if "type" in example and ("O" in example["type"] or None in example["type"]):
                        example.pop("type")
                    annotated_examples.append(example)
            else:
                annotated_examples += batch
            i += batch_size
        assert len(examples) == len(annotated_examples), \
            f"{len(examples)} examples vs. {len(annotated_examples)} annotated examples"
    logging.info(utils.get_label_counter(annotated_examples))
    with open(export_path, mode="w", encoding="utf8") as out_file:
        for example in annotated_examples:
            out_file.write(json.dumps(example))
            out_file.write("\n")


if __name__ == "__main__":
    main()
