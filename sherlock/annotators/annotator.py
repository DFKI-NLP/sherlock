from registrable import Registrable
from typing import List

from sherlock import Document
from sherlock import DocumentProcessor


class Annotator(Registrable, DocumentProcessor):
    """
    Superclass for Annotators that execute a specific task on
    a Document (e.g. Named-Entity-Recognition or Binary
    Relation Classification)
    """

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "Annotator":
        raise NotImplementedError("Annotator must implement 'from_pretrained'.")

    def annotate_documents(self, documents: List[Document]) -> List[Document]:
        return NotImplementedError("Annotator must implement 'annotate_documents'.")

    def annotate_document(self, document: Document) -> Document:
        return self.annotate_documents([document])[0]

