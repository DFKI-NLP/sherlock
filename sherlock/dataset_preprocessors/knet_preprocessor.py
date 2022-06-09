import json
import os
import logging
import argparse
import re

import utils
from relation_types import RELATION_TYPES
from spacy.lang.en import English


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
        # TODO Later, we will probably need these passages to generate negative examples
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
            converted_example = map_knet_label({
                "id": "r/" + utils.generate_example_id(),
                "tokens": word_tokens,
                "label": relation_label,
                "grammar": ["SUBJ", "OBJ"],
                "entities": [[subj_span.start, subj_span.end], [obj_span.start, obj_span.end]],
                # "type": [subj_type, obj_type]
            })
            if converted_example is not None:
                converted_examples.append(converted_example)
    return converted_examples


def map_knet_label(example):
    knet_label = example["label"]
    mapped_label = None

    if knet_label == "CEO":
        mapped_label = "org:top_members/employees"

    # elif knet_label == "CHILD_OF": # child: subj, parent: obj
    #     mapped_label = "per:parents"  # child: subj, parent: obj
    #     example = utils.swap_args(example)
    elif knet_label == "CHILD_OF":  # child: subj, parent: obj
        mapped_label = "per:children"  # child: subj, parent: obj

    elif knet_label == "DATE_FOUNDED":
        mapped_label = "org:founded"

    elif knet_label == "DATE_OF_BIRTH":
        mapped_label = "per:date_of_birth"

    elif knet_label == "DATE_OF_DEATH":
        mapped_label = "per:date_of_death"

    elif knet_label == "EDUCATED_AT":
        mapped_label = "per:schools_attended"

    elif knet_label == "EMPLOYEE_OR_MEMBER_OF":
        mapped_label = "per:employee_of"

    elif knet_label == "FOUNDED_BY":
        mapped_label = "org:founded_by"

    elif knet_label == "HEADQUARTERS":
        # no sensible mapping to
        # "org:city_of_headquarters", "org:country_of_headquarters", "org:stateorprovince_of_headquarters"
        mapped_label = "org:place_of_headquarters"

    if knet_label == "NATIONALITY":
        mapped_label = "per:origin"

    elif knet_label == "POLITICAL_AFFILIATION":
        mapped_label = "per:political_affiliation"

    elif knet_label == "PLACE_OF_BIRTH":
        # no sensible mapping to "per:city_of_birth", "per:country_of_birth", "per:stateorprovince_of_birth"
        mapped_label = "per:place_of_birth"

    elif knet_label == "PLACE_OF_RESIDENCE":
        # no sensible mapping to
        # "per:cities_of_residence", "per:countries_of_residence", "per:stateorprovinces_of_residence"
        mapped_label = "per:places_of_residence"

    elif knet_label == "SPOUSE":
        mapped_label = "per:spouse"

    elif knet_label == "SUBSIDIARY_OF":  # subsidiary: subj, parent: obj
        mapped_label = "org:subsidiaries"  # subsidiary: obj, parent: subj
        example = swap_args(example)
    # elif knet_label == "SUBSIDIARY_OF":
    #     mapped_label = "org:parents"  # subsidiary: subj, parent: obj

    if mapped_label is None:
        return None

    assert mapped_label in RELATION_TYPES
    example["label"] = mapped_label
    return example


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/knowledge-net",
        type=str,
        help="path to directory containing the knowledge net data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/knowledge-net/converted",
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

    spacy_word_splitter = English()
    for split in ["train", "test-no-facts"]:
        split_path = os.path.join(knet_path, split + ".json")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        logging.info("Processing and exporting to %s", split_export_path)
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