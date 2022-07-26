import json
import os
import logging
import argparse

import utils
from docred_preprocessor import map_docred_label


def map_fewrel_label(example):
    # fewrel and docred mostly use the same label set
    return map_docred_label(example)


def fewrel_converter(data, fewrel_rel_info, return_num_discarded=False, spacy_ner_predictor=None):
    num_discarded = 0
    converted_examples = []
    for label, examples in data.items():
        for idx, example in enumerate(examples):
            head_token_positions = example["h"][2][0]
            tail_token_positions = example["t"][2][0]

            subj_start = head_token_positions[0]
            subj_end = head_token_positions[-1]+1
            obj_start = tail_token_positions[0]
            obj_end = tail_token_positions[-1]+1
            converted_example = {
                "id": "r/" + utils.generate_example_id(),
                "tokens": example["tokens"],
                "label": fewrel_rel_info[label][0],
                "grammar": ["SUBJ", "OBJ"],
                "entities": [[subj_start, subj_end], [obj_start, obj_end]]
            }
            converted_examples.append(converted_example)
    if spacy_ner_predictor is not None:
        docs = spacy_ner_predictor([example["tokens"] for example in converted_examples])
        for doc, example in zip(docs, converted_examples):
            subj_start, subj_end = example["entities"][0]
            obj_start, obj_end = example["entities"][0]
            subj_type = utils.get_entity_type(doc, subj_start, subj_end)
            obj_type = utils.get_entity_type(doc, obj_start, obj_end)
            example["type"] = [subj_type, obj_type]
    final_examples = []
    for converted_example in converted_examples:
        converted_example = map_fewrel_label(converted_example)
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
    parser.add_argument(
        "--ner_model_path",
        # default="./models/spacy_trf/model-best",
        type=str,
        help="path to ner model",
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

    spacy_ner_predictor = utils.load_spacy_predictor(args.ner_model_path) if args.ner_model_path else None

    for split in ["train", "val"]:
        split_path = os.path.join(fewrel_path, split + ".json")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        logging.info("Processing and exporting to %s", split_export_path)
        with open(split_path, mode="r", encoding="utf-8") as fewrel_file, \
                open(split_export_path, mode="w", encoding="utf-8") as fewrel_export_file:
            fewrel_data = json.load(fewrel_file)
            converted_examples, num_discarded = fewrel_converter(fewrel_data, fewrel_rel_info,
                                                                 return_num_discarded=True,
                                                                 spacy_ner_predictor=spacy_ner_predictor)

            logging.info(f"{num_discarded} examples were discarded during label mapping")

            final_examples = []
            for example in converted_examples:
                if "type" in example and "O" in example["type"]:
                    logging.warning(f"Examples has erroneous entity types: [{example}]")
                else:
                    final_examples.append(converted_examples)
            logging.info(
                f"Removed {len(converted_examples) - len(final_examples)} examples with erroneous entity types")
            logging.info(f"{len(final_examples)} examples in converted file")
            for conv_example in final_examples:
                fewrel_export_file.write(json.dumps(conv_example))
                fewrel_export_file.write("\n")
            logging.info(utils.get_label_counter(final_examples))


if __name__ == "__main__":
    main()
