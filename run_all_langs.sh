#!/bin/bash
# from http://projects.dfki.uni-kl.de/km-publications/web/ML/core/hpc-doc/docs/slurm-cluster/known-issues/#multithreading-contention
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

for LANG in 'en' 'de' 'es' 'fi' 'fr' 'ru' 'ja' 'zh'
do
 for RUN in {1..5}
 do
  export SEED=$((1337 * RUN))
  export RUN_ID=$RUN
  mkdir -p ./logs/binary_relation_clf/in_language/${LANG}/${RUN}
  mkdir -p ./logs/binary_relation_clf/cross_language/${LANG}/${RUN}
  echo 'Running: ' ./scripts/cluster/batch.sh -p RTXA6000 --gpus=1 ./scripts/cluster/wrapper.sh ./scripts/cluster/binary_relation_clf_${LANG}.sh ' run ' ${RUN_ID}
  ./scripts/cluster/batch.sh -p RTXA6000 --gpus=1 ./scripts/cluster/wrapper.sh ./scripts/cluster/binary_relation_clf_${LANG}.sh 2>&1 > ./logs/binary_relation_clf/in_language/${LANG}/${RUN}/train_evaluate.log
  echo 'Running: ' ./scripts/cluster/batch.sh -p RTXA6000 --gpus=1 ./scripts/cluster/wrapper.sh ./scripts/cluster/binary_relation_clf_${LANG}_mbert.sh ' run ' ${RUN_ID}
  ./scripts/cluster/batch.sh -p RTXA6000 --gpus=1 ./scripts/cluster/wrapper.sh ./scripts/cluster/binary_relation_clf_${LANG}_mbert.sh 2>&1 > ./logs/binary_relation_clf/cross_language/${LANG}/${RUN}/train_evaluate.log
 done
done
