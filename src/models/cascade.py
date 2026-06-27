import joblib

import numpy as np
import pandas as pd

from src.models.svm import EmbeddingClassifier, PosClassifier, StructuralClassifier


class CascadeLP:
    """
    Three-tier link predictor with confidence-based routing.

    Tier 0  Self-loops            — always predict positive (trivial).
    Tier 1  StructuralClassifier  — fast, topology-aware; skip on cold-start.
    Tier 2  PosClassifier         — lightweight semantic; handles cold-start.
    Tier 3  EmbeddingClassifier   — sentence-transformer cosine; for uncertain pairs only.

    A pair escalates to the next tier when the current tier's confidence
    (max predicted probability) falls below the tier's threshold.
    """

    def __init__(self, tier1_threshold: float = 0.8, tier2_threshold: float = 0.7):
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.tier1 = StructuralClassifier()
        self.tier2 = PosClassifier()
        self.tier3 = EmbeddingClassifier()

    def fit(
        self,
        structural: pd.DataFrame,
        pos_features: np.ndarray,
        st_scores: np.ndarray,
        y: np.ndarray,
        pairs: pd.DataFrame,
        n2v: np.ndarray | None = None,
    ):
        """Fit all three tiers independently on the training set (self-loops excluded)."""
        mask = (pairs['id1'].values != pairs['id2'].values)

        print('  Fitting Tier 1 (Structural)...')
        self.tier1.fit(structural[mask], y[mask], n2v[mask] if n2v is not None else None)

        print('  Fitting Tier 2 (POS + RF)...')
        self.tier2.fit(pos_features[mask], y[mask])

        print('  Fitting Tier 3 (Embedding)...')
        self.tier3.fit(st_scores[mask], y[mask])

        return self

    def predict(
        self,
        structural: pd.DataFrame,
        pos_features: np.ndarray,
        st_scores: np.ndarray,
        pairs: pd.DataFrame,
        n2v: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        predictions : (n,) int array — predicted labels
        tier_used   : (n,) int array — which tier made each decision (0-3)
        """
        n = len(pairs)
        predictions = np.zeros(n, dtype=int)
        tier_used   = np.full(n, -1, dtype=int)
        remaining   = np.ones(n, dtype=bool)

        # Tier 0 — self-loops
        self_loop = pairs['id1'].values == pairs['id2'].values
        predictions[self_loop] = 1
        tier_used[self_loop]   = 0
        remaining[self_loop]   = False

        def _route(idx, proba, threshold, tier_id):
            conf      = proba.max(axis=1)
            confident = conf >= threshold
            hit_idx   = idx[confident]
            predictions[hit_idx] = proba[confident].argmax(axis=1)
            tier_used[hit_idx]   = tier_id
            remaining[hit_idx]   = False
            return idx[~confident]          # return indices still undecided

        # Tier 1 — structural
        idx = np.where(remaining)[0]
        if len(idx):
            proba1    = self.tier1.predict_proba(structural.iloc[idx], n2v[idx] if n2v is not None else None)
            idx       = _route(idx, proba1, self.tier1_threshold, 1)

        # Tier 2 — POS
        if len(idx):
            proba2    = self.tier2.predict_proba(pos_features[idx])
            idx       = _route(idx, proba2, self.tier2_threshold, 2)

        # Tier 3 — embedding (handles all remaining; no threshold needed)
        if len(idx):
            predictions[idx] = self.tier3.predict(st_scores[idx])
            tier_used[idx]   = 3

        return predictions, tier_used

    def tier_stats(self, tier_used: np.ndarray) -> dict:
        total = len(tier_used)
        return {
            f'tier{t}': {'n': int((tier_used == t).sum()),
                         'pct': float((tier_used == t).mean() * 100)}
            for t in range(4)
        }

    def save(self, path: str): joblib.dump(self, path)

    @classmethod
    def load(cls, path: str): return joblib.load(path)
