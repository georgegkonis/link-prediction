import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class StructuralClassifier:
    """Logistic Regression on structural heuristic features (CN, Jaccard, AA, PA)."""

    FEATURES = ['cn', 'jaccard', 'adamic_adar', 'pref_attach']

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42, n_jobs=-1)

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.FEATURES].fillna(0).values

    def fit(self, structural: pd.DataFrame, y: np.ndarray):
        self.clf.fit(self.scaler.fit_transform(self._X(structural)), y)
        return self

    def predict_proba(self, structural: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(self._X(structural)))

    def predict(self, structural: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(structural).argmax(axis=1)

    def save(self, path: str): joblib.dump(self, path)

    @classmethod
    def load(cls, path: str): return joblib.load(path)


class TfidfClassifier:
    """Logistic Regression on TF-IDF cosine similarity score."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

    def fit(self, scores: np.ndarray, y: np.ndarray):
        self.clf.fit(self.scaler.fit_transform(scores.reshape(-1, 1)), y)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(scores.reshape(-1, 1)))

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.predict_proba(scores).argmax(axis=1)

    def save(self, path: str): joblib.dump(self, path)

    @classmethod
    def load(cls, path: str): return joblib.load(path)


class PosClassifier:
    """Random Forest on concatenated POS frequency features (72-dim)."""

    def __init__(self, n_estimators: int = 200, n_jobs: int = -1):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs,
            random_state=42, class_weight='balanced',
        )

    def fit(self, pos_features: np.ndarray, y: np.ndarray):
        self.clf.fit(pos_features, y)
        return self

    def predict_proba(self, pos_features: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(pos_features)

    def predict(self, pos_features: np.ndarray) -> np.ndarray:
        return self.clf.predict(pos_features)

    def save(self, path: str): joblib.dump(self, path)

    @classmethod
    def load(cls, path: str): return joblib.load(path)


class EmbeddingClassifier:
    """Logistic Regression on sentence-transformer cosine similarity score."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

    def fit(self, scores: np.ndarray, y: np.ndarray):
        self.clf.fit(self.scaler.fit_transform(scores.reshape(-1, 1)), y)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(scores.reshape(-1, 1)))

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.predict_proba(scores).argmax(axis=1)

    def save(self, path: str): joblib.dump(self, path)

    @classmethod
    def load(cls, path: str): return joblib.load(path)
