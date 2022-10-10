import json
import gzip
import operator
import argparse
import csv

from tqdm import tqdm
from pathlib import Path
from collections import Counter

from sherlock.dataset_preprocessors.utils import get_entities


def _compute_majority_tag(token, exclude_tags=None) -> (str, float):
    """
    Compute the most frequent tag and its probability in token.ent_dist that is not in exclude_tags.
    Note that exclude_tags only affects the tag selection, not the probability computation.
    Returns None,None if all tags are excluded or ent_dist.values() sums to <= 0.
    :param token:
    :param exclude_tags:
    :return:
    """
    if exclude_tags is None:
        exclude_tags = []
    tag_sum = sum(token["ent_dist"].values())
    ent_dist_copy = dict(token["ent_dist"])
    for ex_tag in exclude_tags:
        if ex_tag in ent_dist_copy:
            ent_dist_copy.pop(ex_tag)
    if len(ent_dist_copy) == 0:
        return None, None
    majority_tag = max(ent_dist_copy.items(), key=operator.itemgetter(1))[0]
    prob = max(ent_dist_copy.values()) / tag_sum
    return majority_tag, prob


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/businesswire/businesswire_20210115_20210912_v2_dedup_token_assembled.jsonlines.gz",
        type=str,
        help="path to businesswire data file",
    )
    parser.add_argument(
        "--export_path",
        default="../../ds/text/businesswire/statistics",
        type=str,
        help="path to save csv files with dataset statistics",
    )
    args = parser.parse_args()
    data_path = Path(args.data_path)
    export_path = Path(args.export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    num_docs = 0
    num_sents = 0
    num_tokens = 0
    num_entity_tokens = 0
    entity_token_dist = Counter()
    num_entities = 0
    entity_dist = Counter()

    print("Processing businesswire data")
    with gzip.open(data_path, mode="r") as f:
        for line in tqdm(f, total=17509):
            ner_labels = []
            entity_tokens = []
            num_docs += 1
            doc = json.loads(line)
            num_sents += len(doc["sents"])
            tokens = doc["tokens"]
            num_tokens += len(tokens)
            for token in tokens:
                majority_tag, _ = _compute_majority_tag(token)
                ner_labels.append(majority_tag)
                if majority_tag != "O":
                    entity_tokens.append(majority_tag)
            num_entity_tokens += len(entity_tokens)
            entity_token_dist.update(entity_tokens)
            entities = get_entities(ner_labels)
            entity_labels = [entity["label"] for entity in entities]
            num_entities += len(entities)
            entity_dist.update(entity_labels)

    # Overview table with number of docs, tokens, entity tokens, entities
    overview_dict = {
        'Dokumente': num_docs,
        'Sätze': num_sents,
        'Tokens': num_tokens,
        'Entity Tokens': num_entity_tokens,
        'Entities': num_entities,
    }

    print("Writing overview statistics to file")
    print(overview_dict)
    with open(export_path.joinpath('overview.csv'), 'w', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(overview_dict.items())

    # Table with entity token distribution
    print("Writing entity token distribution to file")
    print(entity_token_dist)
    with open(export_path.joinpath('entity_token_dist.csv'), 'w', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(entity_token_dist.items())

    # Table with entity distribution
    print("Writing entity distribution to file")
    print(entity_dist)
    with open(export_path.joinpath('entity_dist.csv'), 'w', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(entity_dist.items())


if __name__ == "__main__":
    main()
