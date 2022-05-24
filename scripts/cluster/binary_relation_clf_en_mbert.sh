python ./scripts/run_binary_relation_clf.py \
  --model_type bert \
  --model_name_or_path bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluate_during_training \
  --eval_all_checkpoints \
  --data_dir /netscratch/hennig/data/tacred-mbert/ \
  --cache_dir ./.cache/binary_relation_clf/cross_language/en/${RUN_ID} \
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
  --output_dir ./experiments/binary_relation_clf/cross_language/en/${RUN_ID} \
  --dataset_reader tacred \
  --train_file train_en.json \
  --dev_file dev_en.json \
  --test_file test_en.json \
  --predictions_exp_name mbert_en_en_${RUN_ID} \
  --seed ${SEED}
