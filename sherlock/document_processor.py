from typing import List

from sherlock import Document


class DocumentProcessor:
    def __call__(self, documents: List[Document]) -> List[Document]:
        return self.annotate_documents(documents)

    def annotate_documents(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError("DocumentProcessor must implement 'annotate_documents'.")

    def annotate_document(self, document: Document) -> Document:
        return self.annotate_documents([document])[0]
