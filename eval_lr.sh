#!/bin/bash
RUN=11
LANG='fr'
for LRATE in 1e-6 3e-6 5e-6 7e-6 1e-5
do
  export SEED=1337
  #export SEED=$((1337 * RUN))
  export RUN_ID=$RUN
  export LR=$LRATE
  mkdir -p ./logs/binary_relation_clf/in_language/${LANG}/${RUN}
  mkdir -p ./logs/binary_relation_clf/cross_language/${LANG}/${RUN}
  echo 'Running: ' ./scripts/cluster/batch.sh -p RTXA6000 --gpus=1 ./scripts/cluster/wrapper.sh ./scripts/cluster/binary_relation_clf_${LANG}.sh ' run ' ${RUN_ID} ' LR ' ${LR}
  ./scripts/cluster/batch.sh -p RTXA6000 --gpus=1 ./scripts/cluster/wrapper.sh ./scripts/cluster/binary_relation_clf_${LANG}.sh 2>&1 > ./logs/binary_relation_clf/in_language/${LANG}/${RUN}/train_evaluate.log
  RUN=$(($RUN+1))
done
