import json
import argparse
import csv

from pathlib import Path
from collections import Counter


def get_split_stats(file_path):
    num_examples = 0
    num_tokens = 0
    re_labels = []
    with open(file_path, mode="r", encoding="utf8") as f:
        for line in f:
            num_examples += 1
            example = json.loads(line)
            num_tokens += len(example["tokens"])
            re_labels.append(example["label"])
    relation_dist = Counter(re_labels)
    return num_examples, num_tokens, relation_dist


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/UnionizedRelExDataset",
        type=str,
        help="path to unionized relation extraction dataset",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/UnionizedRelExDataset/statistics",
        type=str,
        help="path to save csv files with dataset statistics",
    )
    args = parser.parse_args()
    data_path = Path(args.data_path)
    export_path = Path(args.export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    total_num_examples = 0
    total_num_tokens = 0
    total_relation_dist = Counter()

    split_stats = []

    for split in ["train", "dev", "test"]:
        print(f"Processing {split} data")
        num_examples, num_tokens, relation_dist = get_split_stats(data_path.joinpath(split + ".jsonl"))

        print(relation_dist)
        with open(export_path.joinpath(split + "_relation_distribution.csv"), mode="w", encoding='utf8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(relation_dist.items())

        split_stats.append({
            "Split": split,
            "Beispiele": num_examples,
            "Tokens": num_tokens,
        })

        total_num_examples += num_examples
        total_num_tokens += num_tokens
        total_relation_dist.update(relation_dist)
    split_stats.append({
        "Split": "Gesamt",
        "Beispiele": total_num_examples,
        "Tokens": total_num_tokens,
    })

    print("Overview stats:")
    print(split_stats)
    with open(export_path.joinpath("overview_stats.csv"), mode="w", encoding='utf8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Split", "Beispiele", "Tokens"])
        writer.writeheader()
        writer.writerows(split_stats)

    print("Total relation distribution:")
    print(total_relation_dist)
    with open(export_path.joinpath("total_relation_distribution.csv"), mode="w", encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(total_relation_dist.items())


if __name__ == "__main__":
    main()
