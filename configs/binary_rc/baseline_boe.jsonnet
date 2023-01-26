function (
  num_epochs = 50,
  batch_size = 8,
  lr = 1e-3,
  #do_lower_case = true,
  train_data_path = "/ds/text/patched_tacrev/train.json",
  validation_data_path = "/ds/text/patched_tacrev/dev.json",
  negative_label = "no_relation",
  entity_handling = "mark_entity_append_ner",
  word_dropout = 0.04,
  embedding_dim = 300,
  offset_embedding_dim = 30,
  embedding_dropout = 0.0,
  text_encoder_dropout = 0.5,
  embedding_trainable = false,
  num_classes = 42,
  //offset_type = "relative",
  //text_encoder_activation = null,
  //text_encoder_pooling = "sum",
  text_encoder_projection_dim = null,
  #fp16 = false,
  cuda_device = 0,
  max_instances = null,
  max_len = 100) {

  //local use_offset_embeddings = (offset_embedding_dim != null),
  local use_projection = (text_encoder_projection_dim != null),

  local text_encoder_input_dim = embedding_dim,


  local classifier_feedforward_input_dim = if use_projection then text_encoder_projection_dim else text_encoder_input_dim,


  "dataset_reader": {
    "type": "sherlock",
    "dataset_reader_name": "tacred",
    "feature_converter_name": "binary_rc",
    "tokenizer": {
        "type": "spacy",
    },
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
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
    "f1_average": "micro",
    "ignore_label": negative_label,
    "word_dropout": word_dropout,
    "embedding_dropout": embedding_dropout,
    "encoding_dropout": text_encoder_dropout,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
        "embedding_dim": embedding_dim,
        "trainable": embedding_trainable,
        }
      }
    },

    "text_encoder": {
      "type": "bag_of_embeddings",
      "embedding_dim": text_encoder_input_dim,
      //"pooling": text_encoder_pooling,
      //"activation": text_encoder_activation,
    },
    "classifier_feedforward": {
      "input_dim": classifier_feedforward_input_dim,
      "num_layers": 1,
      "hidden_dims": [num_classes],
      "activations": ["linear"],
      "dropout": [0.0],
    },
  },
    "data_loader": {
      "type": "simple",
      "batch_size": batch_size,
      "shuffle": true,
    },

  "vocabulary": {
    "min_count": {
      "tokens": 2,
    },
  },

  "trainer": {
    "num_epochs": num_epochs,
    "patience": 10,
    "cuda_device": cuda_device,
    "num_serialized_models_to_keep": 1,
    "grad_clipping": 5.0,
    "validation_metric": "+fscore",
    "optimizer": {
      "type": "adam",
      "lr": lr,
    },

    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.9,
      "mode": "max",
      "patience": 1
    },
  }
}
