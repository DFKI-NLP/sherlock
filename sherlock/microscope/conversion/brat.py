from typing import Any, Dict, List, Tuple

from sherlock import Document
from sherlock.microscope.color_palettes import COLOR_PALETTE_ENTITY, COLOR_PALETTE_RELATION


def entity_style(entities: List[Tuple[str, str, List[Tuple[int, int]]]]) -> List[Dict[str, Any]]:
    ent_types = set([ent_type for _, ent_type, _ in entities])
    ent_style = []
    for ent_type, color in zip(ent_types, COLOR_PALETTE_ENTITY):
        ent_style.append(
            {
                "bgColor": color,
                "borderColor": "darken",
                "labels": [ent_type, ent_type],
                "type": ent_type,
            }
        )
    return ent_style


def relation_style(relations: List[Tuple[str, str, List[Tuple[str, str]]]]) -> List[Dict[str, Any]]:
    rel_types = set([rel_type for rel_type, _, _ in relations])
    rel_style = []
    for rel_type, color in zip(rel_types, COLOR_PALETTE_RELATION):
        rel_style.append(
            {
                # 'color': rel_colors[name],
                # 'color': "black",
                # 'dashArray': '1,5',
                "labels": [rel_type, rel_type],
                "type": rel_type,
            }
        )
    return rel_style


def document_to_brat(doc: Document) -> Dict[str, Any]:
    text = doc.text
    entities: List[Tuple[str, str, List[Tuple[int, int]]]] = []
    relations: List[Tuple[str, str, List[Tuple[str, str]]]] = []

    entity_id = 1
    entity_ids = {}
    for mention in sorted(doc.ments, key=lambda m: m.start):
        entity_ids[mention] = entity_id
        start_char = doc.tokens[mention.start].start
        end_char = doc.tokens[mention.end - 1].end
        entities.append((f"T{entity_id}", mention.label, [(start_char, end_char)]))
        entity_id += 1

    relation_id = 1
    for relation in doc.rels:
        head_ment_id = entity_ids[doc.ments[relation.head_idx]]
        tail_ment_id = entity_ids[doc.ments[relation.tail_idx]]
        relations.append(
            (
                f"R{relation_id}",
                relation.label,
                [("", f"T{head_ment_id}"), ("", f"T{tail_ment_id}")],
            )
        )
        relation_id += 1

    return {
        "docData": {"text": text, "entities": entities, "relations": relations},
        "collData": {
            "entity_types": entity_style(entities),
            "relation_types": relation_style(relations),
        },
    }
