import json
import os
import logging
import argparse
import re

from utils import generate_example_id
from spacy.lang.en import English

GLOVE_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
}

TACRED_NEGATIVE_LABEL = 'no_relation'
TACRED_LABELS = [
    TACRED_NEGATIVE_LABEL,
    "org:alternate_names",
    "org:city_of_headquarters",
    "org:country_of_headquarters",
    "org:dissolved",
    "org:founded",
    "org:founded_by",
    "org:member_of",
    "org:members",
    "org:number_of_employees/members",
    "org:parents",
    "org:political/religious_affiliation",
    "org:shareholders",
    "org:stateorprovince_of_headquarters",
    "org:subsidiaries",
    "org:top_members/employees",
    "org:website",
    "per:age",
    "per:alternate_names",
    "per:cause_of_death",
    "per:charges",
    "per:children",
    "per:cities_of_residence",
    "per:city_of_birth",
    "per:city_of_death",
    "per:countries_of_residence",
    "per:country_of_birth",
    "per:country_of_death",
    "per:date_of_birth",
    "per:date_of_death",
    "per:employee_of",
    "per:origin",
    "per:other_family",
    "per:parents",
    "per:religion",
    "per:schools_attended",
    "per:siblings",
    "per:spouse",
    "per:stateorprovince_of_birth",
    "per:stateorprovince_of_death",
    "per:stateorprovinces_of_residence",
    "per:title"
]

KNET_NEGATIVE_LABEL = 'NO_RELATION'
TACRED_RELATIONS_NOT_IN_KNOWLEDGENET = [
    "org:alternate_names",
    "org:dissolved",
    "org:shareholders",
    "org:website",
    "org:number_of_employees/members",
    "org:member_of",  # companies/orgs members of larger confederations
    "org:members",  # companies/orgs members of larger confederations
    "org:political/religious_affiliation",  # TODO: compare against "POLITICAL_AFFILIATION"
    "per:age",
    "per:alternate_names",
    "per:charges",
    "per:other_family",
    "per:religion",
    "per:siblings",
    "per:title",
    "per:cause_of_death",
    "per:city_of_death",
    "per:country_of_death",
    "per:stateorprovince_of_death",
]

KNET_LABELS = [
    KNET_NEGATIVE_LABEL,
    "FOUNDED_BY",
    "POLITICAL_AFFILIATION",
    "PLACE_OF_BIRTH",
    "EDUCATED_AT",
    "DATE_OF_DEATH",
    "NATIONALITY",
    "PLACE_OF_RESIDENCE",
    "CHILD_OF",
    "DATE_OF_BIRTH",
    "HEADQUARTERS",
    "DATE_FOUNDED",
    "EMPLOYEE_OR_MEMBER_OF",
    "SPOUSE",
    "SUBSIDIARY_OF",
    "CEO"
]

KNET_LABELS_TO_PROP_IDS = {
    "CHILD_OF": 34,
    "POLITICAL_AFFILIATION": 45,
    "DATE_OF_BIRTH": 15,
    "HEADQUARTERS": 6,
    "EMPLOYEE_OR_MEMBER_OF": 3,
    "DATE_FOUNDED": 5,
    "EDUCATED_AT": 9,
    "FOUNDED_BY": 2,
    "PLACE_OF_RESIDENCE": 11,
    "DATE_OF_DEATH": 14,
    "CEO": 4,
    "NATIONALITY": 10,
    "PLACE_OF_BIRTH": 12,
    "SPOUSE": 25,
    "SUBSIDIARY_OF": 1
}


def remove_contiguous_whitespaces(text):
    # +1 to account for regular whitespace at the beginning
    contiguous_whitespaces_indices = [(m.start(0) + 1, m.end(0)) for m in re.finditer('  +', text)]
    cleaned_text = re.sub(" +", " ", text)
    return cleaned_text, contiguous_whitespaces_indices


def fix_char_index(char_index, contiguous_whitespaces_indices):
    new_char_index = char_index
    offset = 0
    for ws_start, ws_end in contiguous_whitespaces_indices:
        if char_index >= ws_end:
            offset = offset + (ws_end - ws_start)
    new_char_index -= offset
    return new_char_index


def knowledge_net_converter(example, word_splitter):
    converted_examples = []
    for passage in example["passages"]:
        # Skip passages without facts right away
        # Later, we will probably need these passages to generate negative examples
        if len(passage["facts"]) == 0:
            continue

        text = passage["passageText"]
        cleaned_text, contiguous_ws_indices = remove_contiguous_whitespaces(text)
        doc = word_splitter(cleaned_text)
        word_tokens = [t.text for t in doc]

        passage_start = passage["passageStart"]

        for fact in passage["facts"]:
            subj_start = fix_char_index(fact["subjectStart"] - passage_start, contiguous_ws_indices)
            subj_end = fix_char_index(fact["subjectEnd"] - passage_start, contiguous_ws_indices)
            obj_start = fix_char_index(fact["objectStart"] - passage_start, contiguous_ws_indices)
            obj_end = fix_char_index(fact["objectEnd"] - passage_start, contiguous_ws_indices)
            assert cleaned_text[subj_start:subj_end] == re.sub(" +", " ", fact["subjectText"]), \
                f"Mismatch: " \
                f"<{cleaned_text[subj_start:subj_end]}> vs. <{re.sub(' +', ' ', fact['subjectText'])}>"
            assert cleaned_text[obj_start:obj_end] == re.sub(" +", " ", fact["objectText"]), \
                f"Mismatch: " \
                f"<{cleaned_text[obj_start:obj_end]}> vs. <{re.sub(' +', ' ', fact['objectText'])}>"
            # Get exclusive token spans from char spans
            subj_span = doc.char_span(subj_start, subj_end, alignment_mode="expand")
            obj_span = doc.char_span(obj_start, obj_end, alignment_mode="expand")

            relation_label = fact["humanReadable"].split(">")[1][2:]
            converted_examples.append({
                "id": "r/" + generate_example_id(),
                "tokens": word_tokens,
                "label": relation_label,
                "grammar": ["SUBJ", "OBJ"],
                "entities": [[subj_span.start, subj_span.end], [obj_span.start, obj_span.end]],
                # "type": [subj_type, obj_type]
            })
    return converted_examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="./ds/text/knowledge-net",
        type=str,
        help="path to directory containing the knowledge net data files",
    )
    parser.add_argument(
        "--export_path",
        default="./ds/text/knowledge-net/converted",
        type=str,
        help="path to directory where the converted files should be saved",
    )
    args = parser.parse_args()

    knet_path = args.data_path
    export_path = args.export_path
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    for split in ["train", "test-no-facts"]:
        split_path = os.path.join(knet_path, split + ".json")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        logging.info("Processing and exporting to %s", split_export_path)
        spacy_word_splitter = English()
        with open(split_path, mode="r", encoding="utf-8") as knet_file, \
                open(split_export_path, mode="w", encoding="utf-8") as export_knet_file:
            for line in knet_file.readlines():
                json_data = json.loads(line)
                converted_examples = knowledge_net_converter(json_data, spacy_word_splitter)
                for conv_example in converted_examples:
                    export_knet_file.write(json.dumps(conv_example))
                    export_knet_file.write("\n")


if __name__ == "__main__":
    main()
