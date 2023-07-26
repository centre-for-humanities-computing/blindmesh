from typing import Any, Iterable, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class ClassifierFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        classifier: ClassifierMixin,
        classes: Iterable[Any],
        kind: Literal["positive", "negative"] = "positive",
        freeze: bool = True,
    ):
        self.classifier = classifier
        self.classes = list(classes)
        self.kind = kind
        if self.kind not in ["positive", "negative"]:
            raise ValueError(
                "The filter either has to be positive or negative."
            )
        self.freeze = freeze

    def fit(self, X: list[str], y=None):
        if not self.freeze:
            self.classifier.fit(X, y)
        return self

    def transform(self, X: list[str]) -> list[str]:
        predictions = self.classifier.predict(X)
        mask = np.isin(predictions, self.classes)
        passes = mask if self.kind == "positive" else ~mask
        return [text for p, text in zip(passes, X) if p]
