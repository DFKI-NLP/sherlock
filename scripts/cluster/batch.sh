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

srun -K --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=$IMAGE \
--container-workdir=$WORKDIR \
--ntasks=1 \
--nodes=1 \
$*
