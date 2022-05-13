# 🕵️ Sherlock

<p align="center">
    <!--a href="https://circleci.com/gh/ChristophAlt/sherlock">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a-->
    <a href="https://github.com/psf/black">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>

<h3 align="center">
<p>State-of-the-art Information Extraction
</h3>

Tested with Python 3.8.

Download and install Sherlock:
```bash
git clone git@github.com:DFKI-NLP/sherlock.git
cd sherlock
pip install .
```

# AllenNLP-based

The most straighforward approach to use and test a model is using allennlp:

To train, use the [AllenNLP CLI](https://guide.allennlp.org/training-and-prediction#2). This requires you to setup a [config file](https://guide.allennlp.org/using-config-files). This project includes two example configurations
in the [configs](configs) folder:
* Bert-based transformer for Relation Classification:
[transformer.jsonnet](configs/binary_rc/transformer.jsonnet)
* CNN for Relation Classification:
[cnn.jsonnet](configs/binary_rc/cnn.jsonnet)

To train them you can train the models via:
```bash
# transformer
allennlp train configs/binary_rc/transformer.jsonnet -f -s <serialization dir>

# cnn
allennlp train configs/binary_rc/cnn.jsonnet -f -s <serialization dir>
```

To evaluate the model it is expected that you model.tar.gz file in
the [archive format from AllenNLP](https://docs.allennlp.org/main/api/models/archival/). Now
you have two options:

```bash
# evaluation script
python ./scripts/eval_binary_relation_clf_allennlp.py \
  --eval_data_path <PATH TO EVAL DATA> \
  --test_data_path <PATH TO TEST DATA> \
  --do_eval \
  --do_predict \
  --eval_all_checkpoints \
  --per_gpu_batch_size 8 \
  --output_dir <SERIALIZATION DIR or PATH TO ARCHIVE> \
  --overwrite_results

# allennlp cli
allennlp evaluate <PATH TO ARCHIVE> <PATH TO EVAL DATA> \
  --cuda-device 0 \
  --batch-size 8 \
```

## Configs

The crux of the configs lies in the `dataset_reader` and `model` section.

### dataset_reader

The [dataset_reader](sherlock/allennlp/sherlock_dataset_reader.py) for AllenNLP is a patch-together of the [dataset_reader](sherlock/dataset_readers/) from sherlock and the [feature_converter](sherlock/feature_converters/) from sherlock.

It inherits from `allennlp.data.DatasetReader` and its name ("type") is `"sherlock"`. It accepts a dataset_reader_name,
which must be a registered sherlock-dataset_reader and dataset_reader_kwargs to initialize the dataset_reader with correct arguments.
The same happens for the `feature_converter`. Besides that, it takes
the standart arguments that a AllenNLP-DatasetReader takes.
For more details look into the documentation of the [sherlock_dataset_reader](sherlock/allennlp/sherlock_dataset_reader.py).

### model

The [models](sherlock/allennlp/models/) directory contains the models which
can be used as of now. Because of dependency-injection you can produce quite
a lot with these models already: whereby the [transformer](sherlock/allennp/models/relation_classification/transformer_relation_classifier) model is
limited to a certain type of (bert-like) transformers, the [basic_relation_classifier](sherlock/allennlp/models/relation_classification/basic_relation_classifier.py) can handle anything which fits into the schema of "embedder" -> "encoder" -> "classifier" (yes, theoretically transformer based models too).

*For the transformers module it is important to give it the correct tokenizer keyword arguments, in this case `additional_special_tokens`, as it uses those to rescale its embedding dimension. There did not seem another generic and clean way to do this.*

# Huggingface-based

The original repo was written only with the `transformers` library support.
Although it is possible to use `transformers` models via AllenNLP, Sherlock v2
still supports using the older codebase:

## Named-entity recognition

For example, to train a NER model on the TACRED dataset:


```bash
./scripts/run_ner.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
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
  --output_dir <OUTPUT DIR> \
  --cache_dir <CACHE DIR>
```

## Relation classification

For example, to train a RC model on the TACRED dataset:


```bash
./scripts/run_binary_relation_clf.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
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
  --output_dir <OUTPUT DIR> \
  --cache_dir <CACHE DIR>
```

# Tests

Tests are located in the directory `tests`. To run them, being in the root directory call:
```
py.test
```
or
```
pytest -sv
```
To call a specific test specify testfile and use `-k` flag:
```
pytest tests/feature_converters/token_classification_test.py -sv -k "truncate"
```

# Installation issues

* Using `python==3.9` the installation of `tokenizers` needed for `transformers`
  may fails. Install Rust manually:
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
  [link1](https://www.rust-lang.org/tools/install),
  [link2](https://github.com/huggingface/transformers/issues/2831#issuecomment-600141935)
* If you are not using `conda>=4.10` the installation of `jsonnet` may fail.
  Install it manually: `conda install -c conda-forge jsonnet`
