import os
import json

from tests import FIXTURES_ROOT

# import spacy

from sherlock.document import Token, Span, Entity, Relation, Document


def _doc_from_tacred(example):
    tokens = example["token"]
    ner = example["stanford_ner"]
    pos = example["stanford_pos"]
    dep = example["stanford_deprel"]
    dep_head = example["stanford_head"]

    head_start, head_end = example["subj_start"], example["subj_end"] + 1
    tail_start, tail_end = example["obj_start"], example["obj_end"] + 1
    text = " ".join(tokens)

    doc = Document(guid=example["id"], text=text)

    start_offset = 0
    for idx, token in enumerate(tokens):
        end_offset = start_offset + len(token)
        doc.tokens.append(Token(doc=doc,
                                start=start_offset,
                                end=end_offset,
                                lemma=token,
                                pos=pos[idx],
                                tag=pos[idx],
                                dep=dep[idx],
                                dep_head=dep_head[idx],
                                ent_type=ner[idx]))
        # increment offset because of whitespace
        start_offset = end_offset + 1

    doc.sents = [Span(doc=doc, start=0, end=len(tokens))]
    doc.ments = [Span(doc=doc, start=head_start, end=head_end, label=example["subj_type"]),
                 Span(doc=doc, start=tail_start, end=tail_end, label=example["obj_type"])]
    doc.ents = [Entity(doc=doc, mentions_indices=[0], label=" ".join(tokens[head_start: head_end])),
                Entity(doc=doc, mentions_indices=[1], label=" ".join(tokens[tail_start: tail_end]))]
    doc.rels = [Relation(doc=doc,
                         head_idx=0,
                         tail_idx=1,
                         label=example["relation"])]
    return doc


# def test_create_document_from_spacy():
#     nlp = spacy.load("en_core_web_sm")

#     with open(os.path.join(FIXTURES_ROOT, "tacred.json"), "r") as f:
#         dataset = json.load(f)

#     text = " ".join(dataset[0]["token"])
#     spacy_doc = nlp(text)

#     doc = Document.from_spacy(guid="0", doc=spacy_doc)

#     assert len(doc.tokens) == 26
#     assert doc.tokens[0].text == "At"
#     assert doc.tokens[-1].text == "."
#     assert doc.guid == "0"


def test_document_to_dict():
    with open(os.path.join(FIXTURES_ROOT, "tacred.json"), "r") as f:
        dataset = json.load(f)

    doc = _doc_from_tacred(dataset[0])

    with open(os.path.join(FIXTURES_ROOT, "doc.json"), "r") as f:
        expected_dict = json.load(f)

    assert doc.to_dict() == expected_dict


def test_document_from_dict():
    with open(os.path.join(FIXTURES_ROOT, "doc.json"), "r") as f:
        dct = json.load(f)

    with open(os.path.join(FIXTURES_ROOT, "tacred.json"), "r") as f:
        dataset = json.load(f)

    expected_doc = _doc_from_tacred(dataset[0])

    doc = Document.from_dict(dct)

    assert doc == expected_doc

    assert len(doc.rels) == 1
