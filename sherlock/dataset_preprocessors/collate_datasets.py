#!/usr/bin/python
# -*- coding: utf8 -*-
import utils
import json
import random
import logging
from pathlib import Path


def load_examples(file_path, filter_examples=False):
    examples = []
    with open(file_path, mode="r", encoding="utf8") as f:
        for line in f.readlines():
            example = json.loads(line)
            if not filter_examples or (filter_examples and "type" in example):
                examples.append(example)
    return examples


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Collect all the examples
    collated_examples = []

    docred_path = Path(f"../../ds/text/DocRED/converted")
    collated_examples += load_examples(docred_path.joinpath("train_annotated.jsonl"))

    fewrel_path = Path(f"../../ds/text/fewrel/converted")
    collated_examples += load_examples(fewrel_path.joinpath("train.jsonl"))
    collated_examples += load_examples(fewrel_path.joinpath("val.jsonl"))

    gids_path = Path(f"../../ds/text/gids/converted")
    collated_examples += load_examples(gids_path.joinpath("train.jsonl"))
    collated_examples += load_examples(gids_path.joinpath("dev.jsonl"))
    collated_examples += load_examples(gids_path.joinpath("test.jsonl"))

    kbp37_path = Path(f"../../ds/text/kbp37/converted")
    collated_examples += load_examples(kbp37_path.joinpath("train.jsonl"))
    collated_examples += load_examples(kbp37_path.joinpath("dev.jsonl"))
    collated_examples += load_examples(kbp37_path.joinpath("test.jsonl"))

    knet_path = Path(f"../../ds/text/knowledge-net/converted")
    collated_examples += load_examples(knet_path.joinpath("train.jsonl"))

    plass_product_path = Path(f"../../ds/text/plass-corpus/product-corpus-v3-20190618/converted")
    collated_examples += load_examples(plass_product_path.joinpath("train.jsonl"))
    collated_examples += load_examples(plass_product_path.joinpath("dev.jsonl"))
    collated_examples += load_examples(plass_product_path.joinpath("test.jsonl"))

    plass_sdw_path = Path(f"../../ds/text/plass-corpus/sdw-iter02/converted")
    collated_examples += load_examples(plass_sdw_path.joinpath("train.jsonl"))
    collated_examples += load_examples(plass_sdw_path.joinpath("dev.jsonl"))
    collated_examples += load_examples(plass_sdw_path.joinpath("test.jsonl"))

    smiler_product_path = Path(f"../../ds/text/smiler/converted")
    collated_examples += load_examples(smiler_product_path.joinpath("en-small_corpora_train.jsonl"))
    collated_examples += load_examples(smiler_product_path.joinpath("en-small_corpora_test.jsonl"))

    tacrev_path = Path(f"../../ds/text/tacrev/converted")
    collated_examples += load_examples(tacrev_path.joinpath("train.jsonl"))
    collated_examples += load_examples(tacrev_path.joinpath("dev.jsonl"))
    collated_examples += load_examples(tacrev_path.joinpath("test.jsonl"))

    # Shuffle the examples and create train, dev, test split
    random.seed(42)
    random.shuffle(collated_examples)

    split_1 = int(0.8 * len(collated_examples))
    split_2 = int(0.9 * len(collated_examples))
    train = collated_examples[:split_1]
    dev = collated_examples[split_1:split_2]
    test = collated_examples[split_2:]

    # Write data to files
    export_path = Path(f"../../ds/text/collated")
    export_path.mkdir(parents=True, exist_ok=True)

    for data, export_name in zip([train, dev, test], ["train", "dev", "test"]):
        logging.info(f"Exporting {export_name} data")
        logging.info(utils.get_label_counter(data))
        with open(export_path.joinpath(export_name + ".jsonl"), mode="w", encoding="utf8") as f:
            for example in data:
                f.write(json.dumps(example))
                f.write("\n")


if __name__ == "__main__":
    main()
