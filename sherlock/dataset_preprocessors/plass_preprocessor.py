import json
import os
import logging
import argparse

from relation_types import RELATION_TYPES


def map_plass_label(example):
    plass_label = example["label"]
    mapped_label = None

    if plass_label in RELATION_TYPES:
        mapped_label = plass_label
    elif plass_label == "CompanyProvidesProduct":  # (organization, product)
        mapped_label = "org:product_or_technology_or_service"
    elif plass_label == "Disaster":  # (location, disaster_type)
        mapped_label = "loc:event_or_disaster"
    elif plass_label == "CompanyFacility":  # (organization, location)
        mapped_label = "org:facility_or_location"
    elif plass_label == "CompanyFinancialEvent":    # (organization, financial_event)
        mapped_label = "org:fin_event"
    elif plass_label == "CompanyCustomer":  # (organization, organization)
        mapped_label = "org:customer"
    # elif plass_label == "Identity":
    #     mapped_label = "org:identity"

    if mapped_label is None:
        return None

    assert mapped_label in RELATION_TYPES
    example["label"] = mapped_label
    return example


def plass_converter(data):
    converted_examples = []
    for example in data:
        label = example["relation"]
        entities = [[example["subj_start"], example["subj_end"]+1], [example["obj_start"], example["obj_end"]+1]]
        # TODO ner mapping
        subj_type = example["subj_type"]
        obj_type = example["obj_type"]
        ent_type = [subj_type, obj_type]
        converted_example = map_plass_label({
            "id": example["id"],
            "tokens": example["tokens"],
            "label": label,
            "grammar": ["SUBJ", "OBJ"],
            "entities": entities,
            "type": ent_type
        })
        if converted_example is not None:
            converted_examples.append(converted_example)
    return converted_examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/plass-corpus/product-corpus-v3-20190618",    # sdw-iter02
        type=str,
        help="path to directory containing the plass data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/plass-corpus/product-corpus-v3-20190618/converted",    # sdw-iter02
        type=str,
        help="path to directory where the converted files should be saved",
    )
    args = parser.parse_args()

    plass_path = args.data_path
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
        split_path = os.path.join(plass_path, split + ".jsonl")
        logging.info("Reading %s", split_path)
        split_export_path = os.path.join(export_path, split + ".jsonl")
        logging.info("Processing and exporting to %s", split_export_path)
        with open(split_path, mode="r", encoding="utf-8") as plass_file, \
                open(split_export_path, mode="w", encoding="utf-8") as plass_export_file:
            plass_data = json.load(plass_file)
            converted_examples = plass_converter(plass_data)
            for conv_example in converted_examples:
                plass_export_file.write(json.dumps(conv_example))
                plass_export_file.write("\n")


if __name__ == "__main__":
    main()
