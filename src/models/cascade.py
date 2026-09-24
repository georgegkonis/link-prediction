import joblib

import numpy as np
import pandas as pd

from src.models.svm import EmbeddingClassifier, PosClassifier, StructuralClassifier

_HEURISTIC_COLS = ['cn', 'jaccard', 'adamic_adar', 'pref_attach']


def cold_start_from_structural(structural: pd.DataFrame) -> np.ndarray:
    """
    Boolean mask — True where a pair carries no structural signal at all.

    ``compute_heuristics`` writes exactly 0.0 across all four heuristics when
    either endpoint is absent from the graph, while any pair with both endpoints
    present has ``pref_attach = deg(u) * deg(v) >= 1`` (every node in a graph
    built from an edge list has degree >= 1). An all-zero row therefore
    identifies a cold-start pair unambiguously, which lets CascadeLP route on
    cold-start without being handed the graph.

    This agrees with ``metrics.cold_start_mask`` for graphs without self-loops.
    Zero common neighbours is a separate, broader population: both endpoints
    can still be observed and carry degree or Adamic-Adar evidence.

    Returns all-False if the frame does not carry the heuristic columns.
    """
    if not all(c in structural.columns for c in _HEURISTIC_COLS):
        return np.zeros(len(structural), dtype=bool)
    # NaN marks a self-loop, which Tier 0 has already resolved — never cold-start.
    return (structural[_HEURISTIC_COLS].fillna(1.0) == 0).all(axis=1).to_numpy()


class CascadeLP:
    """
    Three-tier link predictor with confidence-based routing.

    Tier 0  Self-loops            — always predict positive (trivial).
    Tier 1  StructuralClassifier  — fast, topology-aware; skip on cold-start.
    Tier 2  PosClassifier         — lightweight semantic; handles cold-start.
    Tier 3  EmbeddingClassifier   — sentence-transformer cosine; for uncertain pairs only.

    A pair escalates to the next tier when the current tier's confidence
    (max predicted probability) falls below the tier's threshold. Cold-start
    pairs (either endpoint absent from the graph) bypass Tier 1 entirely and
    enter the cascade at Tier 2, since their structural features are all zero
    and carry no signal to be confident about.
    """

    def __init__(self, tier1_threshold: float = 0.8, tier2_threshold: float = 0.7,
                 random_state: int = 42):
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.tier1 = StructuralClassifier(random_state=random_state)
        self.tier2 = PosClassifier(random_state=random_state)
        self.tier3 = EmbeddingClassifier(random_state=random_state)

    def fit(
        self,
        structural: pd.DataFrame,
        pos_features: np.ndarray,
        st_scores: np.ndarray,
        y: np.ndarray,
        pairs: pd.DataFrame,
    ):
        """Fit all three tiers independently on the training set (self-loops excluded)."""
        mask = (pairs['id1'].values != pairs['id2'].values)

        print('  Fitting Tier 1 (Structural)...')
        self.tier1.fit(structural[mask], y[mask])

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
        cold_start: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        cold_start : (n,) bool array, optional
            Pairs to route past Tier 1. Derived from the structural features via
            ``cold_start_from_structural`` when not supplied.

        Returns
        -------
        predictions : (n,) int array   — predicted labels
        tier_used   : (n,) int array   — which tier made each decision (0-3)
        scores      : (n,) float array — P(y=1) at whichever tier resolved the
                      pair (1.0 for Tier-0 self-loops), for AUC-ROC/confidence
                      analysis
        """
        return self._predict(
            structural,
            pairs,
            pos_provider=lambda idx, _: pos_features[idx],
            embedding_provider=lambda idx, _: st_scores[idx],
            cold_start=cold_start,
        )

    def predict_lazy(
        self,
        structural: pd.DataFrame,
        pairs: pd.DataFrame,
        pos_provider,
        embedding_provider,
        cold_start: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict while requesting semantic features only for unresolved pairs.

        Each provider receives a pair frame in original row order. The POS
        provider must return a two-dimensional feature matrix; the embedding
        provider returns one similarity score per pair.
        """
        return self._predict(
            structural,
            pairs,
            pos_provider=lambda _idx, subset: pos_provider(subset),
            embedding_provider=lambda _idx, subset: embedding_provider(subset),
            cold_start=cold_start,
        )

    def _predict(
        self,
        structural: pd.DataFrame,
        pairs: pd.DataFrame,
        pos_provider,
        embedding_provider,
        cold_start: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(pairs)
        predictions = np.zeros(n, dtype=int)
        tier_used   = np.full(n, -1, dtype=int)
        scores      = np.zeros(n, dtype=float)
        remaining   = np.ones(n, dtype=bool)

        # Tier 0 — self-loops
        self_loop = pairs['id1'].values == pairs['id2'].values
        predictions[self_loop] = 1
        tier_used[self_loop]   = 0
        scores[self_loop]      = 1.0
        remaining[self_loop]   = False

        def _route(idx, proba, threshold, tier_id):
            conf      = proba.max(axis=1)
            confident = conf >= threshold
            hit_idx   = idx[confident]
            predictions[hit_idx] = proba[confident].argmax(axis=1)
            tier_used[hit_idx]   = tier_id
            scores[hit_idx]      = proba[confident][:, 1]
            remaining[hit_idx]   = False
            return idx[~confident]          # return indices still undecided

        if cold_start is None:
            cold_start = cold_start_from_structural(structural)
        else:
            cold_start = np.asarray(cold_start, dtype=bool)
            if len(cold_start) != n:
                raise ValueError(f'cold_start has length {len(cold_start)}, expected {n}')

        # Tier 1 — structural (cold-start pairs bypass it: all-zero features)
        idx = np.where(remaining & ~cold_start)[0]
        if len(idx):
            proba1    = self.tier1.predict_proba(structural.iloc[idx])
            idx       = _route(idx, proba1, self.tier1_threshold, 1)

        # Cold-start pairs enter the cascade here, alongside Tier-1 escalations.
        idx = np.sort(np.concatenate([idx, np.where(remaining & cold_start)[0]]))

        # Tier 2 — POS
        if len(idx):
            pos_features = np.asarray(pos_provider(idx, pairs.iloc[idx]))
            if len(pos_features) != len(idx):
                raise ValueError('POS provider returned the wrong number of rows')
            proba2    = self.tier2.predict_proba(pos_features)
            idx       = _route(idx, proba2, self.tier2_threshold, 2)

        # Tier 3 — embedding (handles all remaining; no threshold needed)
        if len(idx):
            st_scores = np.asarray(embedding_provider(idx, pairs.iloc[idx]))
            if len(st_scores) != len(idx):
                raise ValueError('Embedding provider returned the wrong number of rows')
            proba3 = self.tier3.predict_proba(st_scores)
            predictions[idx] = proba3.argmax(axis=1)
            tier_used[idx]   = 3
            scores[idx]      = proba3[:, 1]

        return predictions, tier_used, scores

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
