import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import pipeline


class ZeroShotScorer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        classes: list[str],
        model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    ):
        self.classes = classes
        self.n_classes = len(classes)
        self.model_name = model_name
        self.pipe = pipeline(model=model_name)

    def fit(self, X, y=None):
        pass

    def transform(self, X: list[str]) -> np.ndarray:
        n_texts = len(X)
        res = np.empty((n_texts, self.n_classes))
        for i_doc, text in enumerate(X):
            out = self.pipe(text, candidate_labels=self.classes)
            label_to_score = dict(zip(out["labels"], out["scores"]))
            for i_class, label in enumerate(self.classes):
                res[i_doc, i_class] = label_to_score[label]
