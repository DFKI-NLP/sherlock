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
import shutil
from collections import Counter
from typing import Dict, List, Union, Tuple, Optional, Any

import numpy as np
import torch
from tqdm import tqdm

import allennlp
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary, Instance, DatasetReader
from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from allennlp.data.tokenizers import (
    Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer)
from allennlp.data.token_indexers import (
    TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer)
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules import FeedForward
from allennlp.nn.util import move_to_device
from allennlp.training.learning_rate_schedulers import LinearWithWarmup
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
from allennlp.training import GradientDescentTrainer
from allennlp.training.util import evaluate as evaluateAllennlp
from allennlp.training import Checkpointer

from sherlock.allennlp import SherlockDatasetReader
from sherlock.allennlp.models.relation_classification import BasicRelationClassifier
from sherlock.allennlp.models.relation_classification import TransformerRelationClassifier
from sherlock.dataset_readers import TacredDatasetReader
from sherlock.metrics import compute_f1
from sherlock.tasks import IETask


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_chache_data(
    args,
    dataset_reader: allennlp.data.DatasetReader,
    split: str,
    return_dataset: bool=False,
) -> Union[SimpleDataLoader, Tuple[SimpleDataLoader, List[Instance]]]:
    """Returns Dataloader with unindexed Instances. Optionally returns
    dataset for e.g. vocabulary creation."""

    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process
        # the dataset, and the others will use the cache
        torch.distributed.barrier()

    if not os.path.isdir(args.cache_dir):
        os.mkdir(args.cache_dir)

    # chache_file name has to be sensitive to differences in dataloading.
    cache_file = os.path.join(
        args.cache_dir,
        f"cached_rc_{split}_{os.path.basename(args.model_name_or_path)}"
        + f"_{args.max_seq_length}_{args.entity_handling}"
        + f"_{args.add_inverse_relations}_{args.do_lower_case}"
        + f"_{args.tokenizer_name}",
    )

    # Batch size
    if split == "train":
        batch_size = args.per_gpu_train_batch_size
    elif split == "dev" or split in "validation":
        batch_size = args.per_gpu_eval_batch_size
    elif split == "test":
        batch_size = args.per_gpu_eval_batch_size

    if os.path.exists(cache_file) and not args.overwrite_cache:
        logger.info(
            f"Loading features for split {split} from cached file {cache_file}",
        )
        dataset: List[Instance] = torch.load(cache_file)
    else:
        # TODO: paths as direct paths in args
        if split == "train":
            path_to_data = os.path.join(args.data_dir, "train.json")
        elif split == "dev" or split in "validation":
            path_to_data = os.path.join(args.data_dir, "dev.json")
        elif split == "test":
            path_to_data = os.path.join(args.data_dir, "test.json")

        dataset: List[Instance] = list(dataset_reader.read(path_to_data))
        if args.local_rank in [-1, 0]:
            torch.save(dataset, cache_file)


    if args.local_rank == 0:
        torch.distributed.barrier()

    data_loader = SimpleDataLoader(
        dataset,
        batch_size=batch_size,
    )

    if return_dataset:
        return data_loader, dataset
    return data_loader


def evaluate(
    args,
    eval_dataloader,
    model,
    filename=None,
):
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

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
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--entity_handling",
        type=str,
        default="mask_entity",
        choices=["mark_entity", "mark_entity_append_ner", "mask_entity", "mask_entity_append_text"],
    )
    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run predictions on the test set."
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
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
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.no_cuda:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            logger.warn("Not using cuda although it is available.")
        args.n_gpu = 0
    elif args.local_rank == -1:
        # Set index, because if not, allennlp crashes ¯\_(ツ)_/¯
        if torch.cuda.is_available():
            device = torch.device("cuda", index=0)
        else:
            device = torch.device("cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", index=args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        str(args.device),
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    logger.info("Evaluation parameters %s", args)

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

    # Load everything from archive
    archive = load_archive(
        args.archive_path,
        cuda_device=-1 if args.device == torch.device("cpu") else args.device,
    )
    dataset_reader = archive.validation_dataset_reader
    model = archive.model


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        # Load data
        valid_data_loader = load_and_chache_data(args, dataset_reader, "dev")
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
        logger.info(f"Evaluating archived model")
        evaluateAllennlp(model, valid_data_loader, args.device)
        for checkpoint in checkpoints:
            logger.info(f"Evaluating {checkpoint.split('/')[-1]}")
            splitted = checkpoint.split("_")
            epoch_and_step = \
                f"{splitted[-2]}_{splitted[-1]}" if len(checkpoints) > 1 else ""

            # load checkpoint model
            state_dict = torch.load(checkpoint, map_location=args.device)
            model.load_state_dict(state_dict)

            result = evaluateAllennlp(model, valid_data_loader, args.device)
            result = {f"{k}_{epoch_and_step}": v for k, v in result.items()}
            results.update(result)


    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:

        # Load data
        test_data_loader = load_and_chache_data(args, dataset_reader, "test")
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
