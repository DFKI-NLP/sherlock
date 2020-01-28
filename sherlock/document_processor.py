from typing import List

from sherlock import Document


class DocumentProcessor:
    def __call__(self, documents: List[Document]) -> List[Document]:
        return self.predict_documents(documents)

    def predict_documents(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError("DocumentProcessor must implement 'predict_documents'.")

    def predict_document(self, document: Document) -> Document:
        return self.predict_documents([document])[0]
