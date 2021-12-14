#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array 1-20%4
#SBATCH --job-name test
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-cpu=24G
#SBATCH --partition RTXA6000

username="$USER"
IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.07-py3.sqsh
WORKDIR="`pwd`"

srun -K --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=$IMAGE \
--container-workdir=$WORKDIR \
./scripts/cluster/wrapper.sh \
python ./scripts/run_binary_relation_clf.py \
  --model_type bert \
  --model_name_or_path bert-case-uncased \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluate_during_training \
  --eval_all_checkpoints \
  --do_lower_case \
  --data_dir /ds/text/tacred/data/json/ \
  --cache_dir ./.cache \
  --save_steps 8500 \
  --logging_steps 8500 \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size=8 \
  --per_gpu_train_batch_size=8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --overwrite_cache \
  --overwrite_output_dir \
  --entity_handling mark_entity_append_ner \
  --output_dir ./experiments