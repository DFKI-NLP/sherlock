#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array 1-20%4
#SBATCH --job-name test
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-cpu=24G
#SBATCH --partition RTXA6000

username="$USER"
IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh
WORKDIR="`pwd`"

# cpus-per-task - from http://projects.dfki.uni-kl.de/km-publications/web/ML/core/hpc-doc/docs/slurm-cluster/known-issues/#multithreading-contention

srun 2>&1 -K --container-mounts=/netscratch/$USER:/netscratch/$USER,/netscratch/$USER/.cache_slurm:/root/.cache,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=$IMAGE \
--container-workdir=$WORKDIR \
--ntasks=1 \
--nodes=1 \
--cpus-per-task=1 \
$*
