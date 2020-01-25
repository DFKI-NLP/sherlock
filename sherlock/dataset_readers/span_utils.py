# taken from https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_utils/span_utils.py

from typing import Callable, List, Set, Tuple, TypeVar, Optional
import warnings

from sherlock.document import Token


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return " ".join(self.tag_sequence)


T = TypeVar("T", str, Token)
TypedStringSpan = Tuple[str, Tuple[int, int]]


def to_bioul(tag_sequence: List[str], encoding: str = "IOB1") -> List[str]:
    """
    Given a tag sequence encoded with IOB1 labels, recode to BIOUL.

    In the IOB1 scheme, I is a token inside a span, O is a token outside
    a span and B is the beginning of span immediately following another
    span of the same type.

    In the BIO scheme, I is a token inside a span, O is a token outside
    a span and B is the beginning of a span.

    # Parameters

    tag_sequence : `List[str]`, required.
        The tag sequence encoded in IOB1, e.g. ["I-PER", "I-PER", "O"].
    encoding : `str`, optional, (default = `IOB1`).
        The encoding type to convert from. Must be either "IOB1" or "BIO".

    # Returns

    bioul_sequence : `List[str]`
        The tag sequence encoded in IOB1, e.g. ["B-PER", "L-PER", "O"].
    """
    if encoding not in {"IOB1", "BIO"}:
        raise ValueError(f"Invalid encoding {encoding} passed to 'to_bioul'.")

    def replace_label(full_label, new_label):
        # example: full_label = 'I-PER', new_label = 'U', returns 'U-PER'
        parts = list(full_label.partition("-"))
        parts[0] = new_label
        return "".join(parts)

    def pop_replace_append(in_stack, out_stack, new_label):
        # pop the last element from in_stack, replace the label, append
        # to out_stack
        tag = in_stack.pop()
        new_tag = replace_label(tag, new_label)
        out_stack.append(new_tag)

    def process_stack(stack, out_stack):
        # process a stack of labels, add them to out_stack
        if len(stack) == 1:
            # just a U token
            pop_replace_append(stack, out_stack, "U")
        else:
            # need to code as BIL
            recoded_stack = []
            pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                pop_replace_append(stack, recoded_stack, "I")
            pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            out_stack.extend(recoded_stack)

    # Process the tag_sequence one tag at a time, adding spans to a stack,
    # then recode them.
    bioul_sequence = []
    stack: List[str] = []

    for label in tag_sequence:
        # need to make a dict like
        # token = {'token': 'Matt', "labels": {'conll2003': "B-PER"}
        #                   'gold': 'I-PER'}
        # where 'gold' is the raw value from the CoNLL data set

        if label == "O" and len(stack) == 0:
            bioul_sequence.append(label)
        elif label == "O" and len(stack) > 0:
            # need to process the entries on the stack plus this one
            process_stack(stack, bioul_sequence)
            bioul_sequence.append(label)
        elif label[0] == "I":
            # check if the previous type is the same as this one
            # if it is then append to stack
            # otherwise this start a new entity if the type
            # is different
            if len(stack) == 0:
                if encoding == "BIO":
                    raise InvalidTagSequence(tag_sequence)
                stack.append(label)
            else:
                # check if the previous type is the same as this one
                this_type = label.partition("-")[2]
                prev_type = stack[-1].partition("-")[2]
                if this_type == prev_type:
                    stack.append(label)
                else:
                    if encoding == "BIO":
                        raise InvalidTagSequence(tag_sequence)
                    # a new entity
                    process_stack(stack, bioul_sequence)
                    stack.append(label)
        elif label[0] == "B":
            if len(stack) > 0:
                process_stack(stack, bioul_sequence)
            stack.append(label)
        else:
            raise InvalidTagSequence(tag_sequence)

    # process the stack
    if len(stack) > 0:
        process_stack(stack, bioul_sequence)

    return bioul_sequence
