import json
import os
import logging
import argparse


def tacred_converter(data):
    converted_examples = []
    for example in data:
        # TODO relation mapping
        label = example["relation"]
        # TODO argument swapping for inverse relations
        inverse = False
        entities = [[example["subj_start"], example["subj_end"]+1], [example["obj_start"], example["obj_end"]+1]]
        # TODO ner mapping
        subj_type = example["subj_type"]
        obj_type = example["obj_type"]
        ent_type = [subj_type, obj_type]
        if inverse:
            entities[0], entities[1] = entities[1], entities[0]
            ent_type[0], ent_type[1] = ent_type[1], ent_type[0]
        converted_examples.append({
            "id": example["id"],
            "tokens": example["token"],
            "label": label,
            "grammar": ["SUBJ", "OBJ"],
            "entities": entities,
            "type": ent_type
        })
    return converted_examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/tacred/data/json",
        type=str,
        help="path to directory containing the tacred data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/tacred/data/converted",
        type=str,
        help="path to directory where the converted files should be saved",
    )
    args = parser.parse_args()

    tacred_path = args.data_path
    export_path = args.export_path
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    for split in ["train", "dev", "test"]:
        split_path = os.path.join(tacred_path, split + ".json")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        logging.info("Processing and exporting to %s", split_export_path)
        with open(split_path, mode="r", encoding="utf-8") as tacred_file, \
                open(split_export_path, mode="w", encoding="utf-8") as tacred_export_file:
            tacred_data = json.load(tacred_file)
            converted_examples = tacred_converter(tacred_data)
            for conv_example in converted_examples:
                tacred_export_file.write(json.dumps(conv_example))
                tacred_export_file.write("\n")


if __name__ == "__main__":
    main()
