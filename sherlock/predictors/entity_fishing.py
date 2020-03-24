#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 11.03.20
@author: leonhard.hennig@dfki.de
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from sherlock import Document
from sherlock.document import Entity, Mention
from sherlock.predictors import Predictor


logger = logging.getLogger(__name__)


@Predictor.register("entity_fishing")
class EntityFishingPredictor(Predictor):
    def __init__(
        self,
        service_url: str,
        lang: str = "en",
        connect_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
    ) -> None:
        self.service_url = service_url
        self.lang = lang
        self.connect_timeout_ms = connect_timeout_ms
        self.read_timeout_ms = read_timeout_ms

    @classmethod
    def from_pretrained(  # type: ignore
        cls,
        service_url: str,
        lang: str = "en",
        connect_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
    ) -> "Predictor":  # type: ignore
        return cls(service_url, lang, connect_timeout_ms, read_timeout_ms)

    def predict_documents(self, documents: List[Document]) -> List[Document]:
        return [self.predict_document(d) for d in documents]

    def predict_document(self, document: Document) -> Document:
        params = {"text": document.text, "language": {"lang": self.lang}}

        response = requests.post(
            url=self.service_url,
            json=params,
            timeout=(self.connect_timeout_ms, self.read_timeout_ms),
        )
        if response.status_code == 200:
            data = response.json()
            if "entities" in data:
                entities = EntityFishingPredictor._create_entities_from_mentions(
                    document, data["entities"]
                )
                document.ents.extend(entities)
        else:
            raise Exception(
                f"Entity Fishing API returned response code [{response.status_code}], "
                f"message = [{response.reason}], body = [{response.text}]"
            )
        return document

    @staticmethod
    def _create_entities_from_mentions(
        doc: Document, ef_entities: List[Dict[str, Any]]
    ) -> List[Entity]:

        doc_entities: Dict[str, Entity] = {}

        for ef_entity in ef_entities:
            if "wikidataId" in ef_entity:
                mention_text: str = ef_entity["rawName"]
                start = int(ef_entity["offsetStart"])
                end = int(ef_entity["offsetEnd"])
                wikidata_id: str = ef_entity["wikidataId"]
                wikipedia_ext_ref: str = ef_entity[
                    "wikipediaExternalRef"
                ] if "wikipediaExternalRef" in ef_entity else None

                # NOTE: Further useful fields in ef_entity are
                # "nerd_score", "nerd_selection_score", "type" (nerd_type).

                (mention_idx, doc_mention) = EntityFishingPredictor._find_matching_concept(
                    doc, mention_text, start, end
                )
                if doc_mention:
                    if wikidata_id not in doc_entities:
                        doc_entities[wikidata_id] = Entity(
                            doc=doc,
                            mentions_indices=[mention_idx],
                            label=doc_mention.label,
                            ref_ids={"wikidata": wikidata_id},
                        )
                    else:
                        doc_entities[wikidata_id].mentions_indices.append(mention_idx)

                    if wikipedia_ext_ref and "wikipedia" not in doc_entities[wikidata_id].ref_ids:
                        doc_entities[wikidata_id].ref_ids["wikipedia"] = wikipedia_ext_ref

                    # housekeeping. sort mention indices
                    sorted_m_indices = sorted(doc_entities[wikidata_id].mentions_indices)
                    # must clear/extend since dataclass is Frozen
                    # (dataclasses.FrozenInstanceError: cannot assign to field 'mentions_indices')
                    doc_entities[wikidata_id].mentions_indices.clear()
                    doc_entities[wikidata_id].mentions_indices.extend(sorted_m_indices)

        # housekeeping. sort entities by first mention's start offset
        return sorted(list(doc_entities.values()), key=lambda e: e.mentions[0].start)

    @staticmethod
    def _find_matching_concept(
        doc: Document, mention_text: str, start: int, end: int
    ) -> Tuple[int, Optional[Mention]]:
        if doc.ments:
            for (idx, m) in enumerate(doc.ments):
                char_start = doc.tokens[m.start].start
                char_end = doc.tokens[m.end - 1].end
                if (
                    char_start == start
                    and char_end == end
                    and doc.text[char_start:char_end] == mention_text
                ):
                    return idx, m
        return -1, None
