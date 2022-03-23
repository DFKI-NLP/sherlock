# -*- coding: utf8 -*-
from typing import List

from sherlock import Document


class DocumentProcessor:
    def __call__(self, documents: List[Document]) -> List[Document]:
        return self.process_documents(documents)

    def process_documents(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError("DocumentProcessor must implement 'process_documents'.")

    def process_document(self, document: Document) -> Document:
        return self.process_documents([document])[0]
