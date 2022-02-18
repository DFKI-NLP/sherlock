
function (
    num_epochs = 5,
    batch_size = 8,
    lr = 2e-5,
    adam_epsilon = 1e-8,
    weight_decay = 0.0,
    warmup_steps = 0,
    model_name = "bert-base-uncased",
    do_lower_case = true,
    max_length = 128,
    train_data_path = "/home/gkp/opt/dfki/ds/tacred/data/json/train.json",
    validation_data_path = "/home/gkp/opt/dfki/ds/tacred/data/json/dev.json",
    negative_label = "no_relation",
    entity_handling = "mark_entity_append_ner",
    fp16 = false,
    cuda_device = -1,
    max_instances = null,
) {
    local task = "binary_rc",
    local tokenizer_kwargs = {
        "do_lower_case": do_lower_case,
        "additional_special_tokens": if do_lower_case then [
                '[head=organization]', '[head=person]', '[head_end]', '[head_start]', '[tail=cause_of_death]', '[tail=city]', '[tail=country]', '[tail=criminal_charge]', '[tail=date]', '[tail=duration]', '[tail=ideology]', '[tail=location]', '[tail=misc]', '[tail=nationality]', '[tail=number]', '[tail=organization]', '[tail=person]', '[tail=religion]', '[tail=state_or_province]', '[tail=title]', '[tail=url]', '[tail_end]', '[tail_start]'
            ] else [
                '[HEAD=ORGANIZATION]', '[HEAD=PERSON]', '[HEAD_END]', '[HEAD_START]', '[TAIL=CAUSE_OF_DEATH]', '[TAIL=CITY]', '[TAIL=COUNTRY]', '[TAIL=CRIMINAL_CHARGE]', '[TAIL=DATE]', '[TAIL=DURATION]', '[TAIL=IDEOLOGY]', '[TAIL=LOCATION]', '[TAIL=MISC]', '[TAIL=NATIONALITY]', '[TAIL=NUMBER]', '[TAIL=ORGANIZATION]', '[TAIL=PERSON]', '[TAIL=RELIGION]', '[TAIL=STATE_OR_PROVINCE]', '[TAIL=TITLE]', '[TAIL=URL]', '[TAIL_END]', '[TAIL_START]'
            ],
    },
    local parameter_groups = if weight_decay > 0 then [
        [["(?<!LayerNorm\\.)weight",], {"weight_decay": weight_decay}],
        [["bias", "LayerNorm.weight"], {"weight_decay": 0.0}],
    ] else null,

    "dataset_reader": {
        "type": "sherlock",
        "dataset_reader_name": "tacred",
        "task": task,
        "feature_converter_name": "binary_rc",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name,
            "max_length": max_length,
            "tokenizer_kwargs": tokenizer_kwargs,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "max_length": max_length,
                "tokenizer_kwargs": tokenizer_kwargs,
            },
        },
        "max_tokens": max_length,
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
        "type": "transformer_relation_classifier",
        "model_name": model_name,
        "max_length": max_length,
        "ignore_label": negative_label,
        "f1_average": "micro",
        "tokenizer_kwargs": tokenizer_kwargs,
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