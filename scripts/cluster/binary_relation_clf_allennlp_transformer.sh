/run.sh allennlp train -f \
  -s ./experiments/sherlock_transformer/ \
  ./configs/binary_rc/baseline_random_transformer.jsonnet
python ./scripts/eval_binary_relation_clf_allennlp.py \
  --eval_data_path /ds/text/tacred/data/json/dev.json \
  --test_data_path /ds/text/tacred/data/json/test.json \
  --do_eval \
  --do_predict \
  --eval_all_checkpoints \
  --per_gpu_batch_size 8 \
  --output_dir ./experiments/sherlock_transformer \
  --overwrite_results
