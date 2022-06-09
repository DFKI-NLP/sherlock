import json
import os
import logging
import argparse

import utils
from docred_preprocessor import map_doc_red_label


def map_fewrel_label(example):
    # fewrel and docred mostly use the same label set
    return map_doc_red_label(example)


def fewrel_converter(data, fewrel_rel_info):
    converted_examples = []
    for label, examples in data.items():
        for idx, example in enumerate(examples):
            head_token_positions = example["h"][2][0]
            tail_token_positions = example["t"][2][0]

            subj_start = head_token_positions[0]
            subj_end = head_token_positions[-1]
            obj_start = tail_token_positions[0]
            obj_end = tail_token_positions[-1]
            converted_example = map_fewrel_label({
                "id": "r/" + utils.generate_example_id(),
                "tokens": example["tokens"],
                "label": fewrel_rel_info[label][0],
                "grammar": ["SUBJ", "OBJ"],
                "entities": [[subj_start, subj_end+1], [obj_start, obj_end+1]],
                # "type": [subj_type, obj_type]
            })
            if converted_example is not None:
                converted_examples.append(converted_example)
    return converted_examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/fewrel",
        type=str,
        help="path to directory containing the fewrel relation info and data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/fewrel/converted",
        type=str,
        help="path to directory where the converted files should be saved",
    )
    args = parser.parse_args()

    fewrel_path = args.data_path
    export_path = args.export_path
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    rel_info_path = os.path.join(fewrel_path, "pid2name.json")
    logging.info("Reading doc relation info %s", rel_info_path)
    with open(rel_info_path, mode="r", encoding="utf-8") as f:
        fewrel_rel_info = json.load(f)

    for split in ["train", "val"]:
        split_path = os.path.join(fewrel_path, split + ".json")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        logging.info("Processing and exporting to %s", split_export_path)
        with open(split_path, mode="r", encoding="utf-8") as fewrel_file, \
                open(split_export_path, mode="w", encoding="utf-8") as fewrel_export_file:
            fewrel_data = json.load(fewrel_file)
            converted_examples = fewrel_converter(fewrel_data, fewrel_rel_info)
            for conv_example in converted_examples:
                fewrel_export_file.write(json.dumps(conv_example))
                fewrel_export_file.write("\n")


if __name__ == "__main__":
    main()
