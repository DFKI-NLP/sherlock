# 🕵️ Sherlock

<p align="center">
    <a href="https://circleci.com/gh/ChristophAlt/sherlock">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
</p>

<h3 align="center">
<p>State-of-the-art Information Extraction
</h3>

Download and install Sherlock:
```bash
git clone git@github.com:DFKI-NLP/sherlock.git
cd sherlock
pip install .
```

# Named-entity recognition

For example, to train a NER model on the TACRED dataset:


```bash
./scripts/run_ner.py \
  --model_type bert \
  --model_name_or_path bert-case-uncased \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluate_during_training \
  --eval_all_checkpoints \
  --do_lower_case \
  --data_dir <TACRED DIR> \
  --save_steps 8500 \
  --logging_steps 8500 \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size=8 \
  --per_gpu_train_batch_size=8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --overwrite_cache \
  --overwrite_output_dir \
  --output_dir <OUTPUT DIR>
```

# Relation classification

For example, to train a RC model on the TACRED dataset:


```bash
./scripts/run_binary_relation_clf.py \
  --model_type bert \
  --model_name_or_path bert-case-uncased \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluate_during_training \
  --eval_all_checkpoints \
  --do_lower_case \
  --data_dir <TACRED DIR> \
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
  --output_dir <OUTPUT DIR>
```
