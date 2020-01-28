from registrable import Registrable

from sherlock import DocumentProcessor


class Predictor(Registrable, DocumentProcessor):
    @classmethod
    def from_pretrained(cls, **kwargs) -> "Predictor":
        raise NotImplementedError("Predictor must implement 'from_pretrained'.")
