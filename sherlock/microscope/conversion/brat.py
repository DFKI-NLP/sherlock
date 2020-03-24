from typing import Any, Dict, List, Tuple

from sherlock import Document
from sherlock.microscope.color_palettes import (
    COLOR_PALETTE_ENTITY,
    COLOR_PALETTE_RELATION,
    COLOR_PALETTE_EVENT,
)


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


def event_style(triggers: List[Tuple[str, str, List[Tuple[int, int]]]]) -> List[Dict[str, Any]]:
    evt_types = set([evt_type for _, evt_type, _ in triggers])
    evt_style = []
    for evt_type, color in zip(evt_types, COLOR_PALETTE_EVENT):
        evt_style.append(
            {
                "bgColor": color,
                "borderColor": "darken",
                "labels": [evt_type, evt_type],
                "type": evt_type,
            }
        )
    return evt_style


def document_to_brat(doc: Document) -> Dict[str, Any]:
    """
    Converts a document to the embedded brat format documented here:
    https://brat.nlplab.org/embed.html
    """
    text = doc.text
    entities: List[Tuple[str, str, List[Tuple[int, int]]]] = []
    relations: List[Tuple[str, str, List[Tuple[str, str]]]] = []
    triggers: List[Tuple[str, str, List[Tuple[int, int]]]] = []
    events: List[Tuple[str, str, List[Tuple[str, str]]]] = []

    spans_count = 1
    entity_ids = {}
    for mention in sorted(doc.ments, key=lambda m: m.start):
        entity_ids[mention] = entity_id = f"T{spans_count}"
        start_char = doc.tokens[mention.start].start
        end_char = doc.tokens[mention.end - 1].end
        entities.append((entity_id, mention.label, [(start_char, end_char)]))
        spans_count += 1

    relations_count = 1
    for relation in doc.rels:
        head_ment_id = entity_ids[doc.ments[relation.head_idx]]
        tail_ment_id = entity_ids[doc.ments[relation.tail_idx]]
        relations.append(
            (f"R{relations_count}", relation.label, [("", head_ment_id), ("", tail_ment_id)])
        )
        relations_count += 1

    events_count = 1
    for event in doc.events:
        assert event.trigger is not None, "Only events with triggers are supported so far"
        trigger_id = f"T{spans_count}"
        start_char = doc.tokens[event.trigger.start].start
        end_char = doc.tokens[event.trigger.end - 1].end
        triggers.append((trigger_id, event.event_type, [(start_char, end_char)]))
        event_args = [(role, entity_ids[mention]) for role, mention in event.args]
        events.append((f"E{events_count}", trigger_id, event_args))
        spans_count += 1
        events_count += 1

    return {
        "docData": {
            "text": text,
            "entities": entities,
            "relations": relations,
            "triggers": triggers,
            "events": events,
        },
        "collData": {
            "entity_types": entity_style(entities),
            "relation_types": relation_style(relations),
            "event_types": event_style(triggers),
        },
    }
