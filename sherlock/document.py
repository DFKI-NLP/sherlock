from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    @property
    def text(self) -> str:
        return self.doc.text[self.start : self.end]

    def whitespace(self) -> bool:
        return False if self.start == 0 else self.doc.text[self.start - 1] == " "

    # def __str__(self):
    #     return self.text

    def to_dict(self) -> Dict[str, Any]:
        dct = dict(start=self.start, end=self.end)

        for attr in ["lemma", "pos", "tag", "dep", "dep_head", "ent_type"]:
            attr_val = getattr(self, attr)
            if attr_val is not None:
                dct[attr] = attr_val
        return dct

    @classmethod
    def from_dict(cls, doc: "Document", dct: Dict[str, Any]) -> "Token":
        tmp_dct = dict(doc=doc, start=dct["start"], end=dct["end"])

        for attr in ["lemma", "pos", "tag", "dep", "dep_head", "ent_type"]:
            attr_val = dct.get(attr)
            if attr_val is not None:
                tmp_dct[attr] = attr_val
        return cls(**tmp_dct)

    @classmethod
    def from_spacy(cls, doc: Doc, token: SpacyToken):
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
        )


@dataclass
class Span:
    doc: "Document" = field(compare=False, repr=False)
    start: int
    end: int
    label: Optional[str] = None

    @classmethod
    def from_spacy(cls, span):
        return cls(start=span.start, end=span.end, label=span.label_)

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


@dataclass
class Entity:
    doc: "Document" = field(compare=False, repr=False)
    mentions_indices: List[int]
    label: str

    @property
    def mentions(self) -> List[Span]:
        return [self.doc.ments[idx] for idx in self.mentions_indices]

    def to_dict(self):
        return dict(mentions_indices=self.mentions_indices, label=self.label)

    @classmethod
    def from_dict(cls, doc: "Document", dct: Dict[str, Any]) -> "Entity":
        return cls(doc=doc, mentions_indices=dct["mentions_indices"], label=dct["label"])


@dataclass
class Relation:
    doc: "Document" = field(compare=False, repr=False)
    head_idx: int
    tail_idx: int
    label: str

    @property
    def head(self) -> Span:
        return self.doc.ments[self.head_idx]

    @property
    def tail(self) -> Span:
        return self.doc.ments[self.tail_idx]

    def to_dict(self):
        return dict(head_idx=self.head_idx, tail_idx=self.tail_idx, label=self.label)

    @classmethod
    def from_dict(cls, doc: "Document", dct: Dict[str, Any]) -> "Relation":
        return cls(doc=doc, head_idx=dct["head_idx"], tail_idx=dct["tail_idx"], label=dct["label"])


@dataclass
class Document:
    guid: str
    text: str
    tokens: List[Token] = field(default_factory=list)
    sents: List[Span] = field(default_factory=list)
    ments: List[Span] = field(default_factory=list)
    ents: List[Entity] = field(default_factory=list)
    rels: List[Relation] = field(default_factory=list)

    @property
    def is_tokenized(self) -> bool:
        return len(self.tokens) == 0

    @classmethod
    def from_spacy(cls, guid: str, doc: Doc) -> "Document":
        ret_doc = Document(guid=guid, text=doc.text)
        for token in doc:
            ret_doc.tokens.append(Token.from_spacy(ret_doc, token))
        return ret_doc

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            guid=self.guid,
            text=self.text,
            tokens=[token.to_dict() for token in self.tokens],
            sents=[sent.to_dict() for sent in self.sents],
            ments=[ment.to_dict() for ment in self.ments],
            ents=[ent.to_dict() for ent in self.ents],
            rels=[rel.to_dict() for rel in self.rels],
        )

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "Document":
        doc = Document(guid=dct["guid"], text=dct["text"])
        doc.tokens = [Token.from_dict(doc, token) for token in dct["tokens"]]
        doc.sents = [Span.from_dict(doc, sent) for sent in dct["sents"]]
        doc.ments = [Span.from_dict(doc, ment) for ment in dct["ments"]]
        doc.ents = [Entity.from_dict(doc, ent) for ent in dct["ents"]]
        doc.rels = [Relation.from_dict(doc, rel) for rel in dct["rels"]]
        return doc
