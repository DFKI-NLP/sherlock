#!/usr/bin/python
# -*- coding: utf8 -*-
import utils
import argparse
import json
import random
import logging
from pathlib import Path


def load_examples(file_path, filter_examples=False, id_prefix=None):
    examples = []
    with open(file_path, mode="r", encoding="utf8") as f:
        for line in f.readlines():
            example = json.loads(line)
            if id_prefix is not None:
                example["id"] = f"{id_prefix}_{example['id']}"
            if not filter_examples or (filter_examples and "type" in example):
                examples.append(example)
    return examples

def get_train_test_split(data, test_size=0.2, train_size=0.8, check_tokens_identity=False, shuffle=True):
    assert train_size + test_size == 1.0
    split = int(train_size * len(data))
    if shuffle:
        random.shuffle(data)
    train = data[:split]
    test = data[split:]
    if check_tokens_identity:
        """
            Used for datasets such as DocRED, KnowledgeNet that do not have a train, dev, test split and
            contain multiple relation annotations for the same sentence, resulting in multiple examples
            with the same tokens. Those examples should be put in the same split to avoid data leakage.
        """
        # Move examples from train to test split if the test split contains examples with the same tokens
        train_token_seqs = [" ".join(example["tokens"]) for example in train]
        test_token_seqs = [" ".join(example["tokens"]) for example in test]
        indices = []
        for i, token_seq in enumerate(train_token_seqs):
            if token_seq in test_token_seqs:
                indices.append(i)
        for i in reversed(indices):
            test.append(train.pop(i))
        train_size = len(train) / (len(train) + len(test))
        test_size = 1.0 - train_size
        logging.info(f"Moved {len(indices)} examples from the train split to the test split to prevent data leakage.\n"
                     f"This resulted in a {train_size} train - {test_size} test - split.")
    return train, test


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text",
        type=str,
        help="path to directory containing the data files",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/collated",
        type=str,
        help="path to directory where the collated files should be saved",
    )
    parser.add_argument(
        "--merge_splits",
        default=False,
        action="store_true",
        help="Merge all the train, dev, test files and perform split to the merged dataset",
    )
    parser.add_argument(
        "--check_tokens_identity",
        default=False,
        action="store_true",
        help="Check for identical token sequences across splits and prevent data leakage",
    )
    args = parser.parse_args()
    data_path = Path(args.data_path)
    export_path = Path(args.export_path)
    merge_splits = args.merge_splits
    check_tokens_identity = args.check_tokens_identity

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    random.seed(42)

    docred_path = data_path.joinpath("DocRED/converted")
    fewrel_path = data_path.joinpath("fewrel/converted")
    gids_path = data_path.joinpath("gids/converted")
    kbp37_path = data_path.joinpath("kbp37/converted")
    knet_path = data_path.joinpath("knowledge-net/converted")
    plass_sdw_path = data_path.joinpath("plass-corpus/sdw-iter02/converted")
    plass_product_path = Path(data_path).joinpath("plass-corpus/product-corpus-v3-20190618/converted")
    smiler_product_path = data_path.joinpath("smiler/converted")
    tacrev_path = data_path.joinpath("tacrev/converted")

    if merge_splits:
        collated_examples = []
        collated_examples += load_examples(docred_path.joinpath("train_annotated.jsonl"), id_prefix="DocRED")

        collated_examples += load_examples(fewrel_path.joinpath("train.jsonl"), id_prefix="fewrel")
        collated_examples += load_examples(fewrel_path.joinpath("val.jsonl"), id_prefix="fewrel")

        collated_examples += load_examples(gids_path.joinpath("train.jsonl"), id_prefix="gids")
        collated_examples += load_examples(gids_path.joinpath("dev.jsonl"), id_prefix="gids")
        collated_examples += load_examples(gids_path.joinpath("test.jsonl"), id_prefix="gids")

        collated_examples += load_examples(kbp37_path.joinpath("train.jsonl"), id_prefix="kbp37")
        collated_examples += load_examples(kbp37_path.joinpath("dev.jsonl"), id_prefix="kbp37")
        collated_examples += load_examples(kbp37_path.joinpath("test.jsonl"), id_prefix="kbp37")

        collated_examples += load_examples(knet_path.joinpath("train.jsonl"), id_prefix="knet")

        collated_examples += load_examples(plass_product_path.joinpath("train.jsonl"), id_prefix="plass_product")
        collated_examples += load_examples(plass_product_path.joinpath("dev.jsonl"), id_prefix="plass_product")
        collated_examples += load_examples(plass_product_path.joinpath("test.jsonl"), id_prefix="plass_product")

        collated_examples += load_examples(plass_sdw_path.joinpath("train.jsonl"), id_prefix="plass_sdw")
        collated_examples += load_examples(plass_sdw_path.joinpath("dev.jsonl"), id_prefix="plass_sdw")
        collated_examples += load_examples(plass_sdw_path.joinpath("test.jsonl"), id_prefix="plass_sdw")

        collated_examples += load_examples(smiler_product_path.joinpath("en-small_corpora_train.jsonl"),
                                           id_prefix="smiler")
        collated_examples += load_examples(smiler_product_path.joinpath("en-small_corpora_test.jsonl"),
                                           id_prefix="smiler")

        collated_examples += load_examples(tacrev_path.joinpath("train.jsonl"), id_prefix="tacrev")
        collated_examples += load_examples(tacrev_path.joinpath("dev.jsonl"), id_prefix="tacrev")
        collated_examples += load_examples(tacrev_path.joinpath("test.jsonl"), id_prefix="tacrev")

        train, dev = get_train_test_split(collated_examples, test_size=0.2, train_size=0.8, shuffle=True,
                                                        check_tokens_identity=check_tokens_identity)
        dev, test = get_train_test_split(dev, test_size=0.5, train_size=0.5, shuffle=False,
                                                        check_tokens_identity=check_tokens_identity)
    else:
        train, dev, test = [], [], []

        # If we assume that examples with identical tokens follow each other,
        # then shuffling before splitting results in moving more examples between splits to prevent data leakage
        shuffle = False
        docred_train = load_examples(docred_path.joinpath("train_annotated.jsonl"), id_prefix="DocRED")
        docred_train, docred_dev = get_train_test_split(docred_train, test_size=0.2, train_size=0.8, shuffle=shuffle,
                                                        check_tokens_identity=check_tokens_identity)
        docred_dev, docred_test = get_train_test_split(docred_dev, test_size=0.5, train_size=0.5, shuffle=shuffle,
                                                        check_tokens_identity=check_tokens_identity)

        # we treat fewrel differently because the splits contain different relation types
        fewrel = load_examples(fewrel_path.joinpath("train.jsonl"), id_prefix="fewrel")
        fewrel += load_examples(fewrel_path.joinpath("val.jsonl"), id_prefix="fewrel")
        fewrel_train, fewrel_dev = get_train_test_split(fewrel, test_size=0.2, train_size=0.8, shuffle=True,
                                                         check_tokens_identity=check_tokens_identity)
        fewrel_dev, fewrel_test = get_train_test_split(fewrel_dev, test_size=0.5, train_size=0.5, shuffle=False,
                                                         check_tokens_identity=check_tokens_identity)

        gids_train = load_examples(gids_path.joinpath("train.jsonl"), id_prefix="gids")
        gids_dev = load_examples(gids_path.joinpath("dev.jsonl"), id_prefix="gids")
        gids_test = load_examples(gids_path.joinpath("test.jsonl"), id_prefix="gids")

        kbp37_train = load_examples(kbp37_path.joinpath("train.jsonl"), id_prefix="kbp37")
        kbp37_dev = load_examples(kbp37_path.joinpath("dev.jsonl"), id_prefix="kbp37")
        kbp37_test = load_examples(kbp37_path.joinpath("test.jsonl"), id_prefix="kbp37")

        knet_train = load_examples(knet_path.joinpath("train.jsonl"), id_prefix="knet")
        knet_train, knet_dev = get_train_test_split(knet_train, test_size=0.2, train_size=0.8, shuffle=shuffle,
                                                        check_tokens_identity=check_tokens_identity)
        knet_dev, knet_test = get_train_test_split(knet_dev, test_size=0.5, train_size=0.5, shuffle=shuffle,
                                                        check_tokens_identity=check_tokens_identity)

        plass_product_train = load_examples(plass_product_path.joinpath("train.jsonl"), id_prefix="plass_product")
        plass_product_dev = load_examples(plass_product_path.joinpath("dev.jsonl"), id_prefix="plass_product")
        plass_product_test = load_examples(plass_product_path.joinpath("test.jsonl"), id_prefix="plass_product")

        plass_sdw_train = load_examples(plass_sdw_path.joinpath("train.jsonl"), id_prefix="plass_sdw")
        plass_sdw_dev = load_examples(plass_sdw_path.joinpath("dev.jsonl"), id_prefix="plass_sdw")
        plass_sdw_test = load_examples(plass_sdw_path.joinpath("test.jsonl"), id_prefix="plass_sdw")

        # 19213 training examples - 732 test examples
        smiler_train = load_examples(smiler_product_path.joinpath("en-small_corpora_train.jsonl"),
                                           id_prefix="smiler")
        smiler_test = load_examples(smiler_product_path.joinpath("en-small_corpora_test.jsonl"),
                                           id_prefix="smiler")
        smiler_train, smiler_dev = get_train_test_split(smiler_train, test_size=0.1, train_size=0.9, shuffle=shuffle,
                                                        check_tokens_identity=check_tokens_identity)

        tacrev_train = load_examples(tacrev_path.joinpath("train.jsonl"), id_prefix="tacrev")
        tacrev_dev = load_examples(tacrev_path.joinpath("dev.jsonl"), id_prefix="tacrev")
        tacrev_test = load_examples(tacrev_path.joinpath("test.jsonl"), id_prefix="tacrev")

        for train_split in [
            docred_train, fewrel_train, gids_train, kbp37_train, knet_train, plass_product_train, plass_sdw_train,
            smiler_train, tacrev_train
        ]:
            train += train_split
        random.shuffle(train)
        for dev_split in [
            docred_dev, fewrel_dev, gids_dev, kbp37_dev, knet_dev, plass_product_dev, plass_sdw_dev,
            smiler_dev, tacrev_dev
        ]:
            dev += dev_split
        random.shuffle(dev)
        for test_split in [
            docred_test, fewrel_test, gids_test, kbp37_test, knet_test, plass_product_test, plass_sdw_test,
            smiler_test, tacrev_test
        ]:
            test += test_split
        random.shuffle(test)

    # Write data to files
    export_path.mkdir(parents=True, exist_ok=True)

    for data, export_name in zip([train, dev, test], ["train", "dev", "test"]):
        file_path = export_path.joinpath(export_name + ".jsonl")
        logging.info(f"Exporting {export_name} data ({len(data)} examples) to {file_path}")
        logging.info(utils.get_label_counter(data))
        with open(file_path, mode="w", encoding="utf8") as f:
            for example in data:
                f.write(json.dumps(example))
                f.write("\n")


if __name__ == "__main__":
    main()
