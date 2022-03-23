# -*- coding: utf8 -*-
"""

@date: 01.03.2022
@author: gabriel.kressin@dfki.de
"""

import argparse
from typing import Optional, List

from sherlock.dataset_readers.tacred import TacredDatasetReader
from sherlock.tasks import IETask


def print_additional_tokens(
    file_path: str,
    max_instances: Optional[int] = None,
    as_list: bool = False,
) -> List[str]:

    dataset_reader = TacredDatasetReader(
        max_instances=max_instances,
    )

    additional_tokens = dataset_reader.get_additional_tokens(
        IETask.BINARY_RC, file_path)

    if as_list:
        print(additional_tokens)
    else:
        for additional_token in additional_tokens:
            print(additional_token)

    return additional_tokens


def main():
    parser = argparse.ArgumentParser(
        description="This script gets the additional tokens of a dataset, in"
        + "case that you have to specify them separately somewhere."
    )

    parser.add_argument(
        "file_path",
        default=None,
        type=str,
        help="path to file containing the data determineing the additional tokens.",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=-1,
        help="Only use this number of first instances in dataset (e.g. for debugging).",
    )
    parser.add_argument(
        "--as_list",
        action="store_true",
        help="instead per line, print list of additional tokens.",
    )
    args = parser.parse_args()

    print_additional_tokens(
        args.file_path,
        args.max_instances,
        args.as_list,
    )


if __name__ == "__main__":
    main()