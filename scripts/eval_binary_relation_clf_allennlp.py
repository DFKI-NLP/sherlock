# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from typing import List

import numpy as np
import torch
from tqdm import tqdm

import allennlp
from allennlp.common.checks import ConfigurationError
from allennlp.data import Instance
from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from allennlp.models.archival import load_archive
from allennlp.nn.util import move_to_device
from allennlp.training.util import evaluate as evaluateAllennlp

from sherlock.metrics import compute_f1


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_chache_data(
    args,
    data_path,
    dataset_reader: allennlp.data.DatasetReader,
) -> SimpleDataLoader:
    """Returns Dataloader with unindexed Instances. Optionally returns
    dataset for e.g. vocabulary creation."""

    if args.cache_dir:
        if not os.path.isdir(args.cache_dir):
            os.mkdir(args.cache_dir)
        # chache_file name has to be sensitive to differences in dataloading.
        cache_file = os.path.join(
            args.cache_dir,
            f"cached_rc_{os.path.basename(data_path)}"
            + f"_{args.archive_path}"
        )

        if os.path.exists(cache_file) and not args.overwrite_cache:
            logger.info(
                f"Loading features from cached file {cache_file}",
            )
            dataset: List[Instance] = torch.load(cache_file)
        else:
            dataset: List[Instance] = list(dataset_reader.read(data_path))
            torch.save(dataset, cache_file)
    else:
        dataset: List[Instance] = list(dataset_reader.read(data_path))

    data_loader = SimpleDataLoader(
        dataset,
        batch_size=args.per_gpu_batch_size,
    )

    return data_loader


def evaluate(
    args,
    eval_dataloader,
    model,
    filename=None,
):
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Eval!
    name = filename or ""
    logger.info("***** Running evaluation {} *****".format(name))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    out_label_ids = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        batch = move_to_device(batch, args.device)

        with torch.no_grad():
            outputs = model(**batch)
            tmp_eval_loss, logits = outputs["loss"], outputs["logits"]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        preds.append(logits.detach().cpu().numpy())
        out_label_ids.append(batch["label"].detach().cpu().numpy())

    eval_loss = eval_loss / nb_eval_steps
    preds = np.concatenate(preds, axis=0)
    preds = np.argmax(preds, axis=1)
    out_label_ids = np.concatenate(out_label_ids, axis=0)
    result = compute_f1(preds, out_label_ids)

    logger.info("***** Eval results {} *****".format(name))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    if filename is not None:
        output_eval_file = os.path.join(eval_output_dir, filename)
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result, preds


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--eval_data_path",
        default=None,
        type=str,
        help="path to file containing the evaluation data.",
    )
    parser.add_argument(
        "--test_data_path",
        default=None,
        type=str,
        help="path to file containing the test data",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the models evaluation results and test results will be written."
             + " Also the directory in which the script searches for checkpoints to evaluate.",
    )
    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run predictions on the test set."
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3 and cached features",
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_instances", type=int, default=-1,
        help="Only use this number of first instances in dataset (e.g. for debugging)."
    )
    parser.add_argument(
        "--archive_path", type=str, default=None,
        help="path to archive file of trained model which is evaluated or"
            + "predicted upon."
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Overwrite the evaluation/test results.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    # Setup CUDA
    if args.no_cuda or not torch.cuda.is_available():
        args.device = torch.device("cpu")
        args.cuda_device = -1
        if torch.cuda.is_available():
            logger.warn("Not using cuda although it is available.")
        args.n_gpu = 0
    else:
        # Set index, because if not, allennlp crashes ¯\_(ツ)_/¯
        args.device = torch.device("cuda", index=0)
        args.cuda_device = 0
        args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(f"Device: {str(args.device)}, n_gpu: {args.n_gpu}")
    logger.info("Evaluation parameters %s", args)

    # Set seed
    set_seed(args)

    # Determine right archive path
    if args.archive_path and not os.path.exists(args.archive_path):
        raise ConfigurationError(f"Invalid archive_path: {args.archive_path}")
    else:
        # Try default archive path
        default_archive = os.path.join(args.output_dir, "model.tar.gz")
        if os.path.exists(default_archive):
            args.archive_path = default_archive
        else:
            ConfigurationError(
                f"Cannot find archive at default position: {default_archive}."
                + " please specify archive_path with --archive_path.")

    # Load from archive
    archive = load_archive(
        args.archive_path,
        cuda_device=args.cuda_device,
    )
    dataset_reader = archive.validation_dataset_reader
    model = archive.model


    # Evaluation
    results = {}
    if args.do_eval:

        # Load data
        valid_data_loader = load_and_chache_data(
            args,
            args.eval_data_path,
            dataset_reader,
        )
        valid_data_loader.index_with(model.vocab)

        # Load checkpoints to evaluate
        checkpoints = []
        if args.eval_all_checkpoints:
            search_dir = args.output_dir + "/**/model_state*.th"
            checkpoints = list(
                c for c in sorted(glob.glob(search_dir, recursive=True))
            )

        logger.info(
            f"Evaluate following checkpoints: archived model and {checkpoints}"
        )

        filename_eval_archived_model = os.path.join(
            args.output_dir, "eval_results.txt"
        )
        # Check that evaluation files do not exist already
        if not args.overwrite_results:
            if os.path.exists(filename_eval_archived_model):
                raise ValueError(
                    "Evaluation results for archived model already exist. Use"
                    + " --overwrite_results to overwrite them."
                )
            for checkpoint in checkpoints:
                finalname_eval = f"{os.path.splitext(checkpoint)[0]}_eval_results.txt"
                if os.path.exists(finalname_eval):
                    raise ValueError(
                        f"Evaluation results for checkpoint {checkpoint}"
                        + "already exist, use --overwrite_results to"
                        + "overwrite them."
                    )
        logger.info(f"Evaluating archived model")
        evaluateAllennlp(
            model=model,
            data_loader=valid_data_loader,
            cuda_device=args.cuda_device,
            output_file=filename_eval_archived_model,
        )
        for checkpoint in checkpoints:
            logger.info(f"Evaluating {checkpoint.split('/')[-1]}")
            splitted = checkpoint.split("_")
            epoch_and_step = \
                f"{splitted[-2]}_{splitted[-1]}" if len(checkpoints) > 1 else ""

            # load checkpoint model
            state_dict = torch.load(checkpoint, map_location=args.device)
            model.load_state_dict(state_dict)

            finalname_eval = f"{os.path.splitext(checkpoint)[0]}_eval_results.txt"
            result = evaluateAllennlp(
                model=model,
                data_loader=valid_data_loader,
                cuda_device=args.cuda_device,
                output_file=finalname_eval,
            )
            result = {f"{k}_{epoch_and_step}": v for k, v in result.items()}
            results.update(result)


    # Prediction
    if args.do_predict:

        # Load data
        test_data_loader = load_and_chache_data(
            args,
            args.test_data_path,
            dataset_reader,
        )
        test_data_loader.index_with(model.vocab)

        output_test_results_file = os.path.join(
            args.output_dir, "test_results.txt"
        )
        output_test_predictions_file = os.path.join(
            args.output_dir, "test_predictions.txt"
        )

        result, predictions = evaluate(
            args,
            test_data_loader,
            model,
            filename=output_test_results_file,
        )

        idx_to_label = model.vocab.get_index_to_token_vocabulary("labels")

        predictions = [ idx_to_label[idx] for idx in predictions]
        with open(output_test_predictions_file, "w") as writer:
            for prediction in predictions:
                writer.write(prediction + "\n")

        results.update(result)

    return results


if __name__ == "__main__":
    main()
