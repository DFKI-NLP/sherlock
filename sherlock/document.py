from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from spacy.tokens import Doc
from spacy.tokens import Token as SpacyToken


@dataclass
class Token:
    """
    A simple token representation, keeping track of the token's start and end offset
    in the text it was taken from, POS tag, dependency relation, and similar information.
    Parameters
    ----------
    doc : ``Document``
        The document the token belongs to.
    start : ``int``
        The start character offset of this token in the document text.
    end : ``int``
        The end character offset (exclusive) of this token in the document text.
    lemma : ``str``, optional
        The lemma of this token.
    pos : ``str``, optional
        The coarse-grained part of speech of this token.
    tag : ``str``, optional
        The fine-grained part of speech of this token.
    dep : ``str``, optional
        The dependency relation for this token.
    dep_head : ``int``, optional
        The dependency relation head for this token.
    ent_type : ``str``, optional
        The entity type (i.e., the NER tag) for this token.
    ent_dist: ``dict``, optional
        The distribution of entity types processed by different models for this token.
    """

    doc: "Document" = field(compare=False, repr=False)
    start: int
    end: int
    lemma: Optional[str] = None
    pos: Optional[str] = None
    tag: Optional[str] = None
    dep: Optional[str] = None
    dep_head: Optional[int] = None
    ent_type: Optional[str] = None
    ent_dist: Optional[dict] = None

    @property
    def text(self) -> str:
        return self.doc.text[self.start : self.end]

    @property
    def idx(self) -> int:
        return self.doc.tokens.index(self)

    def whitespace(self) -> bool:
        return False if self.start == 0 else self.doc.text[self.start - 1] == " "

    # def __str__(self):
    #     return self.text

    def to_dict(self) -> Dict[str, Any]:
        dct = dict(start=self.start, end=self.end)

        for attr in ["lemma", "pos", "tag", "dep", "dep_head", "ent_type", "ent_dist"]:
            attr_val = getattr(self, attr)
            if attr_val is not None:
                dct[attr] = attr_val
        return dct

    @classmethod
    def from_dict(cls, doc: "Document", dct: Dict[str, Any]) -> "Token":
        tmp_dct = dict(doc=doc, start=dct["start"], end=dct["end"])

        for attr in ["lemma", "pos", "tag", "dep", "dep_head", "ent_type", "ent_dist"]:
            attr_val = dct.get(attr)
            if attr_val is not None:
                tmp_dct[attr] = attr_val
        return cls(**tmp_dct)

    @classmethod
    def from_spacy(cls, doc: "Document", token: SpacyToken):
        return cls(
            doc=doc,
            start=token.idx,
            end=token.idx + len(token.text),
            lemma=token.lemma_,
            pos=token.pos_,
            tag=token.tag_,
            dep=token.dep_,
            dep_head=token.head.i,
            ent_type=token.ent_type_ or None,
            ent_dist=None
        )


@dataclass(frozen=True)
class Span:
    """
    Representation for an arbitrary Span within a Document.

    A Span is a consecutive sequence of Tokens.
    Saves parent Document, start index, end index and
    a label belonging to the Span.

    Attributes
    ----------
    doc : ``Document``
        Document identifier the Span belongs to
    start : ``int``
        Start index (inclusive)
    end : ``int``
        End index (exclusive)
    label : ``str``, optional
        Saves all sentences within text
    """

    doc: "Document" = field(compare=False, repr=False)
    start: int
    end: int
    label: Optional[str] = None

    @classmethod
    def from_spacy(cls, doc: "Document", span):
        return cls(doc=doc, start=span.start, end=span.end, label=span.label_)

    def to_dict(self):
        dct = dict(start=self.start, end=self.end)
        if self.label:
            dct["label"] = self.label
        return dct

    @classmethod
    def from_dict(cls, doc: "Document", dct: Dict[str, Any]) -> "Span":
        tmp_dct = dict(doc=doc, start=dct["start"], end=dct["end"])
        if "label" in dct:
            tmp_dct["label"] = dct["label"]
        return cls(**tmp_dct)


@dataclass(frozen=True)
class Mention:
    """
    Representation for Mentions of Entities within a Document.

    A Mention is a small group of consecutive Tokens associated to
    an Entity or label (e.g. "Rio de Janeiro").
    Saves parent Document, start index, end index and label where
    a certain Entity is mentioned.

    Attributes
    ----------
    doc : ``Document``
        Document identifier the Entity belongs to
    start : ``int``
        Start index (inclusive)
    end : ``int``
        End index (exclusive)
    label : ``str``
        Saves all sentences within text
    """

    doc: "Document" = field(compare=False, repr=False)
    start: int
    end: int
    label: str

    @classmethod
    def from_spacy(cls, doc: "Document", span):
        return cls(doc=doc, start=span.start, end=span.end, label=span.label_)

    def to_dict(self):
        return dict(start=self.start, end=self.end, label=self.label)

    @property
    def idx(self) -> int:
        return self.doc.ments.index(self)

    @property
    def tokens(self) -> List[Token]:
        return self.doc.tokens[self.start : self.end]

    @property
    def text(self) -> str:
        return self.doc.text[self.tokens[0].start : self.tokens[-1].end]

    @classmethod
    def from_dict(cls, doc: "Document", dct: Dict[str, Any]) -> "Mention":
        return cls(doc=doc, start=dct["start"], end=dct["end"], label=dct["label"])


@dataclass(frozen=True)
class Entity:
    """
    Representation for an Entity class within a Document.

    An entity describes a unique, real-world concept, such as a
    Person, that is referred to by a group of Mentions within this
    document as well as a list of reference ids in external knowledge
    bases, such as Wikidata or Freebase.

    Attributes
    ----------
    doc : ``Document``
        Document identifier the Entity belongs to
    mentions_indices : ``List[int]``
        List of indices of Mentions of this Entitiy within the Document
    label : ``str``
        Label belonging to Entity
    ref_ids : ``Dict[str, Any]``
        I have no idea
    """

    doc: "Document" = field(compare=False, repr=False)
    mentions_indices: List[int]
    label: str
    ref_ids: Dict[str, Any]

    @property
    def idx(self) -> int:
        return self.doc.ents.index(self)

    @property
    def mentions(self) -> List[Mention]:
        return [self.doc.ments[idx] for idx in self.mentions_indices]

    def to_dict(self):
        return dict(mentions_indices=self.mentions_indices, label=self.label, ref_ids=self.ref_ids)

    @classmethod
    def from_dict(cls, doc: "Document", dct: Dict[str, Any]) -> "Entity":
        return cls(
            doc=doc,
            mentions_indices=dct["mentions_indices"],
            label=dct["label"],
            ref_ids=dct["ref_ids"],
        )


@dataclass(frozen=True)
class Relation:
    """
    Representation for a certain Relation within a Document.

    A Relation is a specific instantiation of a relationship between
    a head Mention and a tail Mention within the Document.
    Saves a specific Relation with its label, its head and tail Entity.
    TODO: this is not consistent with Entity:
          Entity has an abstract class (Entity) and its mentions
          separate, whereas every Relation is counted for itself.

    TODO: logits: actual label or only index of transformer label?

    Attributes
    ----------
    doc : ``Document``
        Document identifier the Entity belongs to
    head_idx : ``int``
        Index of Mention which is head Entity of the Relation
    tail_idx : ``int``
        Index of Mention which is tail Entity of the Relation
    label : ``str``
        label belonging to Relation
    logits : ``Dict[str, float]``, optional
        Dictionary containing every label and its predicted logit.
    """

    doc: "Document" = field(compare=False, repr=False)
    head_idx: int
    tail_idx: int
    label: str
    logits: Optional[Dict[str, float]] = None

    @property
    def idx(self) -> int:
        return self.doc.rels.index(self)

    @property
    def head(self) -> Mention:
        return self.doc.ments[self.head_idx]

    @property
    def tail(self) -> Mention:
        return self.doc.ments[self.tail_idx]

    def to_dict(self):
        mention_dict = dict(head_idx=self.head_idx, tail_idx=self.tail_idx, label=self.label)
        if self.logits is not None:
            mention_dict["logits"] = self.logits
        return mention_dict

    @classmethod
    def from_dict(cls, doc: "Document", dct: Dict[str, Any]) -> "Relation":
        return cls(
            doc=doc,
            head_idx=dct["head_idx"],
            tail_idx=dct["tail_idx"],
            label=dct["label"],
            logits=dct.get("logits", None),
        )


@dataclass(frozen=True)
class Event:
    """
    Representation for Events within a Document.

    Saves a single Event.
    TODO: this is not consistent with Entity:
          Entity has an abstract class (Entity) and its mentions
          separate, whereas every Relation is counted for itself.

    Attributes
    ----------
    doc : ``Document``
        Document identifier the Entity belongs to
    event_type : ``str``
        ???
    arg_idx : ``List[Tuple[str, int]]``
        ???
    trigger_idx : ``int``, optional
        ???
    """

    doc: "Document" = field(compare=False, repr=False)
    event_type: str
    arg_idxs: List[Tuple[str, int]]  # tuples of role and mention_idx
    trigger_idx: Optional[int]

    @property
    def idx(self) -> int:
        return self.doc.events.index(self)

    @property
    def args(self) -> List[Tuple[str, Mention]]:
        return [(role, self.doc.ments[arg_idx]) for role, arg_idx in self.arg_idxs]

    @property
    def trigger(self) -> Optional[Mention]:
        if self.trigger_idx is not None:
            return self.doc.ments[self.trigger_idx]
        else:
            return None

    def to_dict(self):
        dct = dict()
        if self.trigger_idx is not None:
            dct["trigger_idx"] = self.trigger_idx
        dct["arg_idxs"] = self.arg_idxs
        dct["event_type"] = self.event_type
        return dct

    @classmethod
    def from_dict(cls, doc: "Document", dct: Dict[str, Any]) -> "Event":
        if "trigger_idx" in dct:
            trigger_idx = dct["trigger_idx"]
        else:
            trigger_idx = None
        return cls(
            doc=doc, event_type=dct["event_type"], arg_idxs=dct["arg_idxs"], trigger_idx=trigger_idx
        )


@dataclass
class Document:
    """
    Representation for a Document

    Main holder for raw text, Token,  Entity, Relation, Mention,
    and Event representations.

    Also saves title, sentences, paragraphs, provenance and
    tokenization status.

    TODO: provenance?

    Attributes
    ----------
    guid : ``str``
        Document identifier
    text : ``str``
        Raw string representation of text in Document.
    tokens : ``List(Token)``
        List of all Tokens within text.
    sents : ``List(Span)``
        List of all sentences within text as Span.
    ments : ``List[Mention]``
        List of all Mentions. Entities and Relations reference this
        list with indices.
    ents : ``List[Entity]``
        List of different Entities in Document. (Not their Mentions!)
    rels : ``List[Relation]``
        List of all Relations in Document. (All instances, not types.)
    events : ``List[Event]``, optional
        List of Events in Document.
    provenance : ``List[Any]``, optional
        ???
    paragraphs : ``List[Span]``, optional
        List of Spans representing Paragraphs.
    title : ``str``, optional
        Document title.
    """

    guid: str
    text: str
    tokens: List[Token] = field(default_factory=list)
    sents: List[Span] = field(default_factory=list)
    ments: List[Mention] = field(default_factory=list)
    ents: List[Entity] = field(default_factory=list)
    rels: List[Relation] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    provenance: Optional[List[Any]] = None
    paragraphs: Optional[List[Span]] = None
    title: Optional[str] = None

    @property
    def is_tokenized(self) -> bool:
        return len(self.tokens) > 0

    @classmethod
    def from_spacy(cls, guid: str, doc: Doc) -> "Document":
        ret_doc = Document(guid=guid, text=doc.text)
        for token in doc:
            ret_doc.tokens.append(Token.from_spacy(ret_doc, token))
        return ret_doc

    def to_dict(self) -> Dict[str, Any]:
        dct = dict(
            guid=self.guid,
            text=self.text,
            tokens=[token.to_dict() for token in self.tokens],
            sents=[sent.to_dict() for sent in self.sents],
            ments=[ment.to_dict() for ment in self.ments],
            ents=[ent.to_dict() for ent in self.ents],
            rels=[rel.to_dict() for rel in self.rels],
        )
        # This is not optimal since explicitly setting events to an empty list
        # is different from not setting any events at all
        if len(self.events) > 0:
            dct["events"] = [evt.to_dict() for evt in self.events]
        if self.provenance is not None:
            dct["provenance"] = self.provenance
        if self.paragraphs is not None:
            dct["paragraphs"] = [par.to_dict() for par in self.paragraphs]
        if self.title is not None:
            dct["title"] = self.title
        return dct

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "Document":
        doc = Document(guid=dct["guid"], text=dct["text"])
        doc.tokens = [Token.from_dict(doc, token) for token in dct["tokens"]]
        doc.sents = [Span.from_dict(doc, sent) for sent in dct["sents"]]
        doc.ments = [Mention.from_dict(doc, ment) for ment in dct["ments"]]
        doc.ents = [Entity.from_dict(doc, ent) for ent in dct["ents"]]
        doc.rels = [Relation.from_dict(doc, rel) for rel in dct["rels"]]
        if "events" in dct:
            doc.events = [Event.from_dict(doc, evt) for evt in dct["events"]]
        if "provenance" in dct:
            doc.provenance = dct["provenance"]
        if "paragraphs" in dct:
            doc.paragraphs = [Span.from_dict(doc, par) for par in dct["paragraphs"]]
        if "title" in dct:
            doc.title = dct["title"]
        return doc
