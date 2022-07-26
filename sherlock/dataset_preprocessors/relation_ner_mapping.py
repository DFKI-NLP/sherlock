import logging
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
        obj_type = "ORG"    # may be a country -> LOC/GPE
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
        obj_type = "ORG"    # TODO check
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
        obj_type = "LOC"    # GPE(?)
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
        obj_type = "LOCATION"
    elif relation_label == "org:member_of":
        subj_type = "ORG"
        obj_type = "ORG"
    elif relation_label == "per:member_of":
        subj_type = "PERSON"
        obj_type = "ORG"
    elif relation_label == "loc:location_of":
        subj_type = "LOC"
        obj_type = "ORG"
        if obj_type not in ["PERSON", "ORG", "MISC", "WORK_OF_ART"]:
            obj_type = "PERSON"     # TODO ambivalent
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
        obj_type = "LOC"    # could be GPE
    elif relation_label == "per:composer":
        subj_type = "PERSON"
        obj_type = "WORK_OF_ART"
    elif relation_label == "per:conflict":
        subj_type = "PERSON"    # TODO often ORG
        obj_type = "EVENT"
    elif relation_label == "loc:contains_location": # TODO check fewrel
        subj_type = "LOC"
        obj_type = "LOC"
    elif relation_label == "loc:country":
        subj_type = "LOC"   # TODO org/loc
        obj_type = "LOC"
    elif relation_label == "loc:country_of_origin":
        subj_type = "LOC"
        if obj_type not in ["MISC", "ORG", "PERSON"]:
            obj_type = "MISC"      # TODO misc/org/per
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
        obj_type = "WORK_OF_ART"    # TODO or MISC?
    elif relation_label == "org:production_company":
        subj_type = "ORG"
        obj_type = "WORK_OF_ART"    # TODO or MISC?
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

    if subj_type is not None and obj_type is not None:
        assert subj_type in NER_TYPES and obj_type in NER_TYPES
    else:
        logging.debug(f"Did not map from [{relation_label}] to any NER labels for subject and object")
    return subj_type, obj_type
