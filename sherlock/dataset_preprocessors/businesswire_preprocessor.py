import argparse
import json
import logging
import sys
from pathlib import Path

from sherlock.dataset_preprocessors.utils import open_file


# Setup logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def businesswire_converter(data):
    # only set the ent_type field and remove ent_dist
    for document in data:
        for idx, token in enumerate(document["tokens"]):
            ent_dist = token.pop("ent_dist")
            majority_ner_label = max(ent_dist, key=ent_dist.get)
            document["tokens"][idx]["ent_type"] = majority_ner_label
        yield document


def process_businesswire(data_path, export_path):
    data_path = Path(data_path)
    export_path = Path(export_path)
    export_path.parent.absolute().mkdir(parents=True, exist_ok=True)
    logger.info("Reading %s", data_path)
    with open_file(data_path, mode="r") as input_file, \
            open_file(export_path, mode="w") as output_file:
        businesswire_data = (json.loads(line) for line in input_file)

        logger.info("Processing and exporting to %s", export_path)
        converted_examples = businesswire_converter(businesswire_data)

        example_counter = 0
        for conv_example in converted_examples:
            output_file.write(json.dumps(conv_example))
            output_file.write("\n")
            example_counter += 1

        logger.info(f"{example_counter} docs in converted file")


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default="../../ds/text/businesswire/businesswire_20210115_20210912_v2_dedup_token_assembled.jsonlines.gz",
        type=str,
        help="path to businesswire file",
    )
    parser.add_argument(
        "--export_path",
        default=
        "../../ds/text/businesswire/converted/businesswire_20210115_20210912_v2_dedup_token_assembled.jsonlines",
        type=str,
        help="path to save converted businesswire file",
    )
    args = parser.parse_args()

    data_path = args.data_path
    export_path = args.export_path

    process_businesswire(data_path, export_path)


if __name__ == "__main__":
    main()
