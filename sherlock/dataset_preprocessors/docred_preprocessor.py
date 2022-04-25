import json
import os
import logging
import argparse

from utils import generate_example_id


def doc_red_converter(example, docred_rel_info):
    labels = example["labels"]
    converted_examples = []
    for idx, label in enumerate(labels):
        # TODO relation type mapping
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
            converted_examples.append({
                "id": "r/" + generate_example_id(),
                "tokens": example["sents"][sent_id],
                "label": rel_type,
                "grammar": ["SUBJ", "OBJ"],
                "entities": [[subj_start, subj_end], [obj_start, obj_end]],
                "type": [subj_type, obj_type]
            })
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
