import json
import gzip
import argparse
import csv

from tqdm import tqdm
from pathlib import Path
from collections import Counter

from sherlock.dataset_preprocessors.utils import get_entities, _compute_majority_tag


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default=
        "../../ds/text/businesswire/annotated/businesswire_20210115_20210912_v2_dedup_token_assembled.jsonlines.gz",
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
    num_relations = 0
    relation_dist = Counter()

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
            relations = doc["rels"]
            num_relations += len(relations)
            relation_labels = [relation["label"] for relation in relations]
            relation_dist.update(relation_labels)

    # Overview table with number of docs, tokens, entity tokens, entities
    overview_dict = {
        'Dokumente': num_docs,
        'Sätze': num_sents,
        'Tokens': num_tokens,
        'Entity Tokens': num_entity_tokens,
        'Entities': num_entities,
        'Relationen': num_relations
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

    # Table with relation distribution
    print("Writing relation distribution to file")
    print(relation_dist)
    with open(export_path.joinpath('relation_dist.csv'), 'w', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(relation_dist.items())


if __name__ == "__main__":
    main()
