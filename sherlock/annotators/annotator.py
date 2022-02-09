# -*- coding: utf8 -*-
"""

@date: 09.02.22
@author: gabriel.kressin@dfki.de
"""
from registrable import Registrable
from typing import List

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
