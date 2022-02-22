
function (
    num_epochs = 5,
    batch_size = 8,
    lr = 2e-5,
    adam_epsilon = 1e-8,
    weight_decay = 0.0,
    warmup_steps = 0,
    do_lower_case = true,
    train_data_path = "../ds/tacred/data/json/train.json",
    validation_data_path = "../ds/tacred/data/json/dev.json",
    negative_label = "no_relation",
    entity_handling = "mark_entity_append_ner",
    word_dropout = 0.04,
    embedding_dropout = 0.0,
    encoding_dropout = 0.0,
    embedding_dim = 300,
    embedding_trainable = false,
    text_encoder_num_filters = 500,
    text_encoder_ngram_filter_sizes = [2, 3, 4, 5],
    num_classes = 42, // not clean
    fp16 = false,
    cuda_device = 0,
    max_instances = 100,
) {
    local parameter_groups = if weight_decay > 0 then [
        [["(?<!LayerNorm\\.)weight",], {"weight_decay": weight_decay}],
        [["bias", "LayerNorm.weight"], {"weight_decay": 0.0}],
    ] else null,
    local text_encoder_input_dim = embedding_dim,
    local classifier_feedforward_input_dim = text_encoder_num_filters * std.length(text_encoder_ngram_filter_sizes),

    "dataset_reader": {
        "type": "sherlock",
        "dataset_reader_name": "tacred",
        "task": "binary_rc",
        "feature_converter_name": "binary_rc",
        "tokenizer": {
            "type": "spacy",
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": false,
            },
        },
        "log_num_input_features": 3,
        "dataset_reader_kwargs": {
            "negative_label_re": negative_label,
        },
        "feature_converter_kwargs": {
            "entity_handling": entity_handling,
        },
        "max_instances": max_instances,
    },

    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,

    "model": {
        "type": "basic_relation_classifier",
        "ignore_label": negative_label,
        "f1_average": "micro",
        "word_dropout": word_dropout,
        "embedding_dropout": embedding_dropout,
        "encoding_dropout": encoding_dropout,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
                    "embedding_dim": embedding_dim,
                    "trainable": embedding_trainable,
                },
            },
        },
        "text_encoder": {
            "type": "cnn",
            "embedding_dim": text_encoder_input_dim,
            "num_filters": text_encoder_num_filters,
            "ngram_filter_sizes": text_encoder_ngram_filter_sizes,
            "conv_layer_activation": "tanh",
        },
        "classifier_feedforward": {
            "input_dim": classifier_feedforward_input_dim,
            "num_layers": 1,
            "hidden_dims": [num_classes],
            "activations": ["linear"],
            "dropout": [0.0],
        },
        "regularizer": {
            "regexes": [
                ["text_encoder.conv_layer_.*weight", {"type": "l2", "alpha": 1e-3}],
            ],
        },
    },
    "data_loader": {
        "type": "simple",
        "batch_size": batch_size,
        "shuffle": true,
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "huggingface_adamw",
            "parameter_groups": parameter_groups,
            "lr": lr,
            "eps": adam_epsilon,
        },
        "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            "warmup_steps": warmup_steps,
        },
        "validation_metric": "+fscore",
        "cuda_device": cuda_device,
        "use_amp": fp16,
    },
}
