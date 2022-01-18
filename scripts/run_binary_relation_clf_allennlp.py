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
from typing import Iterable, List, Union, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import allennlp
from allennlp.data import Vocabulary, Instance
from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from allennlp.data.tokenizers import (
    WhitespaceTokenizer, PretrainedTransformerTokenizer)
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer, PretrainedTransformerIndexer)
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules import FeedForward
from allennlp.training.learning_rate_schedulers import LinearWithWarmup
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
from allennlp.training import GradientDescentTrainer
from allennlp.training.util import evaluate as evaluateAllennlp
from allennlp.training import Checkpointer
from sherlock.models.relation_classification import TransformerRelationClassifier

from sherlock import dataset
from sherlock.dataset import TensorDictDataset
from sherlock.dataset_readers import TacredDatasetReader
from sherlock.dataset_readers import dataset_reader
from sherlock.dataset_readers.dataset_reader import DatasetReader
from sherlock.feature_converters import BinaryRcConverter
from sherlock.metrics import compute_f1
from sherlock.tasks import IETask
from sherlock.models.relation_classification import BasicRelationClassifier


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


def train(
    args,
    train_data_loader,
    valid_data_loader,
    model,
):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.max_steps > 0:
        t_total = args.max_steps
        # TODO: this looks sketchy, gradient_accumulation_steps should be multiplied.
        args.num_train_epochs = (
            args.max_steps // (len(train_data_loader) // args.gradient_accumulation_steps) + 1
        )
    else:
        # t_total is the amount of gradient update/backwards steps
        t_total = len(train_data_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         find_unused_parameters=True,
    #     )

    # Prepare training allennlp
    logger.info("Building Trainer.")
    groups = [
        (["(?<!LayerNorm\.)weight"], {"weight_decay": args.weight_decay}),
        (["bias", "LayerNorm.weight"], {"weight_decay": 0.0}),
    ]
    optimizer = HuggingfaceAdamWOptimizer(
        model_parameters=model.named_parameters(),
        parameter_groups=groups,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )
    scheduler = LinearWithWarmup(
        optimizer=optimizer,
        num_epochs=args.num_train_epochs,
        num_steps_per_epoch=len(train_data_loader) // args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
    )
    # object to decide when to save checkpoints
    checkpointer = Checkpointer(
        serialization_dir=args.output_dir,
        save_every_num_batches=args.save_steps,
        keep_most_recent_by_count=None,
    )
    # object to train model, every device gets its own trainer
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=args.output_dir,
        optimizer=optimizer,
        data_loader=train_data_loader,
        validation_data_loader=valid_data_loader,
        num_epochs=args.num_train_epochs,
        checkpointer=checkpointer,
        learning_rate_scheduler=scheduler,
        cuda_device=-1 if ("cpu" in str(args.device)) else args.device,
        # local_rank=args.local_rank,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.fp16,
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_loader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer.train()
    return


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


def _build_transformers_model(
    args, vocabulary: Vocabulary, weights: Optional[torch.Tensor]) -> Model:
    """Returns Transformers model within AllenNLP framework."""
    return TransformerRelationClassifier(
        vocab=vocabulary,
        model_name=args.model_name_or_path,
        max_length=args.max_seq_length,
        ignore_label=args.negative_label,
        weights=weights,
    )


def _build_basic_model(
    args, vocabulary: Vocabulary, weights: Optional[torch.Tensor]) -> Model:
    """Returns basic AllenNLP model"""

    vocab_size = vocabulary.get_vocab_size()
    label_size = vocabulary.get_vocab_size("labels")

    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=20, num_embeddings=vocab_size)}
    )

    encoder = BagOfEmbeddingsEncoder(embedding_dim=20)

    feedforward = FeedForward(
        input_dim=20,
        num_layers=1,
        hidden_dims=label_size,
        activations=torch.nn.ReLU(),
        dropout=0.5,
    )

    return BasicRelationClassifier(
        vocabulary,
        embedder,
        encoder,
        feedforward,
        ignore_label=args.negative_label,
        weights=weights,
    )



def build_model(
    args, vocabulary: Vocabulary, weights: Optional[torch.Tensor]=None) -> Model:
    """Returns specified Allennlp Model."""

    if args.model_type == "transformers":
        model = _build_transformers_model(args, vocabulary, weights)
    elif args.model_type == "basic":
        model = _build_basic_model(args, vocabulary, weights)
    else:
        raise NotImplementedError(
            f"No Model for model_type: {args.model_type}")
    model.to(args.device)
    return model


def _build_transformers_dataset_reader(args) -> allennlp.data.DatasetReader:
    """Returns appropriate DatasetReader for transformers model."""

    logger.info("Loading Transformer Tokenizer.")
    tokenizer = PretrainedTransformerTokenizer(
        args.tokenizer_name or args.model_name_or_path,
        max_length=args.max_seq_length,
        tokenizer_kwargs={"do_lower_case": args.do_lower_case},
    )

    logger.info("Loading Transformer Indexer.")
    token_indexer = PretrainedTransformerIndexer(
        args.model_name_or_path,
        max_length=args.max_seq_length,
    )

    # Allennlp DatasetReader
    AllennlpDatasetReader = allennlp.data.DatasetReader.by_name("sherlock_reader")
    dataset_reader = AllennlpDatasetReader(
        task="binary_rc",
        dataset_reader_name="tacred",
        feature_converter_name = "binary_rc",
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        max_tokens=args.max_seq_length,
        feature_converter_kwargs={"entity_handling": args.entity_handling},
    )
    return dataset_reader


def _build_basic_dataset_reader(args) -> allennlp.data.DatasetReader:
    """Returns most basic version of a DatasetReader."""

    tokenizer = WhitespaceTokenizer()
    token_indexers = {"tokens": SingleIdTokenIndexer()}

    # Allennlp DatasetReader
    AllennlpDatasetReader = allennlp.data.DatasetReader.by_name("sherlock_reader")
    dataset_reader = AllennlpDatasetReader(
        task="binary_rc",
        dataset_reader_name="tacred",
        feature_converter_name="binary_rc",
        tokenizer=tokenizer,
        token_indexers=token_indexers,
        max_tokens=args.max_seq_length,
        feature_converter_kwargs={"entity_handling": args.entity_handling},
    )
    return dataset_reader


def build_dataset_reader(args) -> allennlp.data.DatasetReader:
    """Returns appropriate DatasetReader for model_type."""

    if args.model_type == "transformers":
        return _build_transformers_dataset_reader(args)
    elif args.model_type == "basic":
        return _build_basic_dataset_reader(args)
    else:
        raise NotImplementedError(
            f"No DatasetReader for model_type: {args.model_type}")


def _reset_output_dir(args, default_vocab_dir) -> None:
    """Deletes old output_dir contents."""

    old_files = (
        glob.glob(os.path.join(args.output_dir, "model_state_*.th"))
        + glob.glob(os.path.join(args.output_dir, "training_state_*.th"))
        + glob.glob(os.path.join(args.output_dir, "metrics_epoch_*.json"))
    )
    for old_file in old_files:
        if os.path.isfile(old_file):
            os.remove(old_file)

    # remove old vocabulary, make sure it is not the same as given in the param
    if (
        os.path.isdir(default_vocab_dir)
        and (
            not args.vocab_dir
            or (
                os.path.realpath(args.vocab_dir)
                != os.path.realpath(default_vocab_dir)
                )
            )
        ):
        shutil.rmtree(default_vocab_dir)


# def evaluate(args, dataset_reader, converter, model, tokenizer, split, prefix=""):
#     eval_output_dir = args.output_dir

#     eval_dataset = load_and_cache_examples(args, dataset_reader, converter, tokenizer, split)

#     if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
#         os.makedirs(eval_output_dir)

#     args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
#     # Note that DistributedSampler samples randomly
#     eval_sampler = (
#         SequentialSampler(eval_dataset)
#         if args.local_rank == -1
#         else DistributedSampler(eval_dataset)
#     )
#     eval_dataloader = DataLoader(
#         eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
#     )

#     # Eval!
#     logger.info("***** Running evaluation {} *****".format(prefix))
#     logger.info("  Num examples = %d", len(eval_dataset))
#     logger.info("  Batch size = %d", args.eval_batch_size)
#     eval_loss = 0.0
#     nb_eval_steps = 0
#     preds = None
#     out_label_ids = None
#     for batch in tqdm(eval_dataloader, desc="Evaluating"):
#         model.eval()
#         batch = {k: t.to(args.device) for k, t in batch.items()}
#         if args.model_type not in ["bert", "xlnet"]:
#             batch["token_type_ids"] = None

#         if args.model_type == "distilbert":
#             del batch["token_type_ids"]

#         with torch.no_grad():
#             outputs = model(**batch)
#             tmp_eval_loss, logits = outputs[:2]

#             eval_loss += tmp_eval_loss.mean().item()
#         nb_eval_steps += 1
#         if preds is None:
#             preds = logits.detach().cpu().numpy()
#             out_label_ids = batch["labels"].detach().cpu().numpy()
#         else:
#             preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
#             out_label_ids = np.append(out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0)

#     eval_loss = eval_loss / nb_eval_steps
#     preds = np.argmax(preds, axis=1)
#     result = compute_f1(preds, out_label_ids)

#     output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
#     with open(output_eval_file, "w") as writer:
#         logger.info("***** Eval results {} *****".format(prefix))
#         for key in sorted(result.keys()):
#             logger.info("  %s = %s", key, str(result[key]))
#             writer.write("%s = %s\n" % (key, str(result[key])))

#     return result, preds


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
        "--model_type",
        default=None,
        type=str,
        required=True,
        choices=["transformers", "basic"],
        help="Model type: ['transformers', 'basic']",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Task-specific parameters
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="If model_type=='transformers': path to pre-trained model or"
        + " shortcut name selected in the transformers hub:"
        + " https://huggingface.co/models",
    )
    parser.add_argument("--negative_label", default="no_relation", type=str)
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
        "--add_inverse_relations",
        action="store_true",
        help="Whether to also add inverse relations to the document.",
    )
    parser.add_argument(
        "--vocab_dir",
        default=None,
        help="Path to directory containing a vocabulary to be used."
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="For Transformers: Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
        help="Where do you want to store the pre-trained models downloaded from s3 and cached features",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--weighted_labels",
        action="store_true",
        help="Weight labels based on their number occurances, useful for"
             + " unbalanced datasets.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=50,
        help="Save checkpoint every X updates steps (every X batches)."
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.no_cuda:
        device = torch.device("cpu", index=0)
        if torch.cuda.is_available():
            logger.warn("Not using cuda although it is available.")
        args.n_gpu = 0
    elif args.local_rank == -1:
        # Set index, because if not, allennlp crashes ¯\_(ツ)_/¯
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu", index=0)
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

    # if args.local_rank not in [-1, 0]:
    #     # Make sure only the first process in distributed training will download model & vocab
    #     torch.distributed.barrier()

    # args.model_type = args.model_type.lower()
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(
    #     args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels
    # )
    # tokenizer = tokenizer_class.from_pretrained(
    #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #     do_lower_case=args.do_lower_case,
    # )
    # model = model_class.from_pretrained(
    #     args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config
    # )


    # additional_tokens = dataset_reader.get_additional_tokens(task=IETask.BINARY_RC)
    # if additional_tokens:
    #     tokenizer.add_tokens(additional_tokens)
    #     model.resize_token_embeddings(len(tokenizer))

    # if args.local_rank == 0:
    #     # Make sure only the first process in distributed training will download model & vocab
    #     torch.distributed.barrier()

    # dataset reader
    logger.info("Loading DatasetReader.")
    dataset_reader = build_dataset_reader(args)

    # vocabulary dir
    default_vocab_dir = os.path.join(args.output_dir, "vocabulary")

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # output_dir is empty or overwrite_output_dir is True, if not there was
        # an error beforehand -> remove previous checkpoints
        _reset_output_dir(args, default_vocab_dir)
        # Load data
        train_data_loader, train_dataset = load_and_chache_data(
            args, dataset_reader, "train", return_dataset=True)
        valid_data_loader = load_and_chache_data(args, dataset_reader, "dev")

        # load vocabulary
        if args.vocab_dir:
            # If given, use a custom vocabulary
            logger.info(f"Loading Vocabulary from {args.vocab_dir}")
            vocabulary = Vocabulary.from_files(args.vocab_dir)
        else:
            # Else Vocabulary from instances
            logger.info("Creating Vocabulary from Dataset.")
            vocabulary = Vocabulary.from_instances(train_dataset)

        if args.model_type == "transformers":
            # PretrainedTransformers have their own vocabulary, extend
            # vocabulary with theirs.
            logger.info(
                "Extending Vocabulary with pretrained Transformer Vocabulary")
            vocabulary_t = Vocabulary.from_pretrained_transformer(
                model_name=args.model_name_or_path,
            )
            vocabulary.extend_from_vocab(vocabulary_t)

        n_labels = vocabulary.get_vocab_size("labels")
        logger.info(
            f"Vocabulary: Tokens: {vocabulary.get_vocab_size()}"
            + f" Labels: {n_labels}"
        )

        train_data_loader.index_with(vocabulary)
        valid_data_loader.index_with(vocabulary)

        if args.weighted_labels:
            # Compute class distributions
            logger.info("Counting Class distribution.")
            counter_classes = Counter()
            for batch in train_data_loader:
                counter_classes.update(batch["label"].tolist())

            # Compute label weights
            message: List[str] = []
            weights = torch.empty((n_labels,))
            for label_id in range(n_labels):
                if counter_classes[label_id] != 0:
                    weights[label_id] = 1 / counter_classes[label_id]
                else:
                    weights[label_id] = 0
                message.append(f"{label_id:2}: {counter_classes[label_id]};")
            weights = (weights / torch.sum(weights))
            weights.to(args.device)
            logger.info(f"Distribution: {' '.join(message)}")
            logger.info(f"Label weights: {weights.tolist()}")
        else:
            weights=None


        # Init Model
        # can only build model here, because vocabulary is needed, and the
        # vocabulary is loaded in only after choosing what to do (train/eval)
        model = build_model(args, vocabulary, weights)

        train(args, train_data_loader, valid_data_loader, model)

    # # Saving best-practices: if you use defaults names for the model,
    # # you can reload it using from_pretrained()
    # if (
    #     args.do_train
    #     and (args.local_rank == -1 or torch.distributed.get_rank() == 0)
    #     and not args.tpu
    # ):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)

        # Save vocabulary if not given
        if not args.vocab_dir:
            vocabulary.save_to_files(default_vocab_dir)

        # Save arguments
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        logger.info(
            f"Saved model checkpoints and vocabulary to {args.output_dir}")

    # Need information beforehand whether vocab_dir!=default_vocab_dir, thus
    # it only can be set here
    if not args.vocab_dir:
        args.vocab_dir = default_vocab_dir

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # Load Vocabulary
        vocabulary = Vocabulary.from_files(args.vocab_dir)

        # Load data
        valid_data_loader = load_and_chache_data(args, dataset_reader, "dev")
        valid_data_loader.index_with(vocabulary)

        # Handle label weights
        if args.weighted_labels:
            # placeholder weights
            weights = torch.empty((vocabulary.get_vocab_size("labels"),))
        else:
            weights = None

        # Init model
        model = build_model(args, vocabulary, weights)

        # Load checkpooints to evaluate
        checkpoints = [os.path.join(args.output_dir, "best.th")]
        if args.eval_all_checkpoints:
            search_dir = args.output_dir + "/**/model_state*.th"
            checkpoints = list(
                c for c in sorted(glob.glob(search_dir, recursive=True))
            )

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
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
        # Load Vocabulary
        vocabulary = Vocabulary.from_files(args.vocab_dir)

        # Load data
        test_data_loader = load_and_chache_data(args, dataset_reader, "test")
        test_data_loader.index_with(vocabulary)

        # Handle label weights
        if args.weighted_labels:
            # placeholder weights
            weights = torch.empty((vocabulary.get_vocab_size("labels"),))
        else:
            weights = None

        # Init model
        model = build_model(args, vocabulary, weights)

        # Load best model
        best_model_path = os.path.join(args.output_dir, "best.th")
        state_dict = torch.load(best_model_path, map_location=args.device)
        model.load_state_dict(state_dict)

        output_test_results_file = os.path.join(
            args.output_dir, "test_results.txt")
        output_test_predictions_file = os.path.join(
            args.output_dir, "test_predictions.txt")

        result = evaluateAllennlp(
            model,
            test_data_loader,
            cuda_device=args.device,
            output_file=output_test_results_file,
            predictions_output_file=output_test_predictions_file,
        )
        # Technically, the predictions are missing in results vs the
        # transformers version of this code TODO: include argmax preds
        results.update(result)

    return results


if __name__ == "__main__":
    main()
