import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


class SvmClassifier:
    """
    RBF-kernel SVM on TF-IDF cosine similarity (Al Hasan et al., 2006-style baseline).
    Trained on a stratified subsample — SVC's O(n^2)-O(n^3) fit cost is
    infeasible on the full 948K training pairs.
    """

    def __init__(self, C: float = 1.0, gamma: str = 'scale',
                 subsample_size: int = 20_000, random_state: int = 42):
        self.subsample_size = subsample_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=random_state)

    def _subsample(self, X: np.ndarray, y: np.ndarray):
        if len(y) <= self.subsample_size:
            return X, y
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=self.subsample_size, stratify=y, random_state=self.random_state)
        return X_sub, y_sub

    def fit(self, scores: np.ndarray, y: np.ndarray):
        X, y = self._subsample(scores.reshape(-1, 1), y)
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(scores.reshape(-1, 1)))

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.predict_proba(scores).argmax(axis=1)

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
