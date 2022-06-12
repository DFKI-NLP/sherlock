from uuid import uuid4
from collections import Counter


def generate_example_id():
    return str(uuid4())


def swap_args(example):
    example["entities"][0], example["entities"][1] = example["entities"][1], example["entities"][0]
    if "type" in example:
        example["type"][0], example["type"][1] = example["type"][1], example["type"][0]
    return example


def get_label_counter(examples):
    labels = [example["label"] for example in examples]
    return dict(Counter(labels))
