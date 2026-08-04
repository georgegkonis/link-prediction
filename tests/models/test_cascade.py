"""Tests for src/models/cascade.py — CascadeLP, the thesis' novel contribution.

Routing is made fully deterministic by replacing the three tier classifiers with
stubs whose `predict_proba` is driven straight off the feature values:

    tier 1  reads column 'p' of the structural frame   → proba = [1-p, p]
    tier 2  reads column 0 of the POS feature matrix    → proba = [1-p, p]
    tier 3  reads the scalar ST score                   → proba = [1-p, p]

Confidence is `max(proba)`, so p=0.95 and p=0.05 are both "confident" while
p=0.5 is not.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.cascade import CascadeLP


# ── stubs ────────────────────────────────────────────────────────────────────

def _proba_from_p(p):
    p = np.asarray(p, dtype=float).ravel()
    return np.column_stack([1.0 - p, p])


class _StubTier1:
    def __init__(self):
        self.calls = []

    def fit(self, structural, y, n2v=None):
        self.calls.append(('fit', len(structural), None if n2v is None else len(n2v)))
        return self

    def predict_proba(self, structural, n2v=None):
        self.calls.append(('predict', list(structural.index)))
        return _proba_from_p(structural['p'].values)


class _StubMatrixTier:
    """Tier 2 — takes a 2-D feature matrix, reads column 0 as P(y=1)."""

    def __init__(self):
        self.calls = []

    def fit(self, X, y):
        self.calls.append(('fit', len(X)))
        return self

    def predict_proba(self, X):
        self.calls.append(('predict', np.asarray(X)[:, 0].tolist()))
        return _proba_from_p(np.asarray(X)[:, 0])


class _StubScalarTier:
    """Tier 3 — takes a 1-D score vector read directly as P(y=1)."""

    def __init__(self):
        self.calls = []

    def fit(self, scores, y):
        self.calls.append(('fit', len(scores)))
        return self

    def predict_proba(self, scores):
        self.calls.append(('predict', np.asarray(scores).tolist()))
        return _proba_from_p(scores)


def _stubbed(tier1_threshold=0.8, tier2_threshold=0.7):
    m = CascadeLP(tier1_threshold=tier1_threshold, tier2_threshold=tier2_threshold)
    m.tier1, m.tier2, m.tier3 = _StubTier1(), _StubMatrixTier(), _StubScalarTier()
    return m


# ── routing scenario ─────────────────────────────────────────────────────────
#
# row  pair    p1     p2     p3    expected
#  0  (5,5)    -      -      -     tier 0, pred 1, score 1.0   (self-loop)
#  1  (1,2)   0.95    -      -     tier 1, pred 1, score 0.95  (conf 0.95 ≥ 0.8)
#  2  (1,3)   0.02    -      -     tier 1, pred 0, score 0.02  (conf 0.98 ≥ 0.8)
#  3  (1,4)   0.50   0.90    -     tier 2, pred 1, score 0.90  (conf 0.90 ≥ 0.7)
#  4  (2,4)   0.60   0.50   0.30   tier 3, pred 0, score 0.30
#  5  (2,5)   0.75   0.35   0.80   tier 3, pred 1, score 0.80  (conf 0.65 < 0.7)

@pytest.fixture
def scenario():
    pairs = pd.DataFrame(
        {'id1': [5, 1, 1, 1, 2, 2], 'id2': [5, 2, 3, 4, 4, 5]},
        index=[100, 101, 102, 103, 104, 105],
    )
    structural = pd.DataFrame(
        {'p': [0.0, 0.95, 0.02, 0.50, 0.60, 0.75]}, index=pairs.index)
    pos = np.zeros((6, 3))
    pos[:, 0] = [0.0, 0.0, 0.0, 0.90, 0.50, 0.35]
    st = np.array([0.0, 0.0, 0.0, 0.0, 0.30, 0.80])
    return pairs, structural, pos, st


def test_tier_routing_is_exact(scenario):
    pairs, structural, pos, st = scenario
    pred, tier, scores = _stubbed().predict(structural, pos, st, pairs)

    assert tier.tolist() == [0, 1, 1, 2, 3, 3]
    assert pred.tolist() == [1, 1, 0, 1, 0, 1]
    assert scores.tolist() == pytest.approx([1.0, 0.95, 0.02, 0.90, 0.30, 0.80])


def test_self_loop_resolves_at_tier_zero_without_touching_any_model():
    pairs = pd.DataFrame({'id1': [9, 9], 'id2': [9, 9]})
    structural = pd.DataFrame({'p': [0.5, 0.5]})
    m = _stubbed()
    pred, tier, scores = m.predict(structural, np.zeros((2, 3)), np.zeros(2), pairs)

    assert tier.tolist() == [0, 0]
    assert pred.tolist() == [1, 1]
    assert scores.tolist() == [1.0, 1.0]
    assert m.tier1.calls == [] and m.tier2.calls == [] and m.tier3.calls == []


def test_confident_negative_also_stops_at_tier_one():
    """Confidence is max(proba), so a confident *negative* must not escalate."""
    pairs = pd.DataFrame({'id1': [1], 'id2': [2]})
    m = _stubbed()
    pred, tier, _ = m.predict(pd.DataFrame({'p': [0.05]}), np.zeros((1, 3)),
                              np.zeros(1), pairs)
    assert tier.tolist() == [1]
    assert pred.tolist() == [0]
    assert m.tier2.calls == [] and m.tier3.calls == []


def test_threshold_boundary_is_inclusive():
    """conf == threshold resolves at that tier (>=, not >)."""
    pairs = pd.DataFrame({'id1': [1, 1], 'id2': [2, 3]})
    m = _stubbed(tier1_threshold=0.8)
    _, tier, _ = m.predict(pd.DataFrame({'p': [0.80, 0.7999]}),
                           np.zeros((2, 3)), np.zeros(2), pairs)
    assert tier.tolist() == [1, 2]


def test_tier_three_is_the_terminal_tier_with_no_threshold():
    """Even a maximally uncertain pair (p=0.5) must be resolved at tier 3."""
    pairs = pd.DataFrame({'id1': [1], 'id2': [2]})
    m = _stubbed()
    pred, tier, scores = m.predict(pd.DataFrame({'p': [0.5]}),
                                   np.full((1, 3), 0.5), np.array([0.5]), pairs)
    assert tier.tolist() == [3]
    assert pred.tolist() == [0]              # argmax ties → class 0
    assert scores.tolist() == [0.5]
    assert -1 not in tier


def test_every_pair_is_assigned_a_tier(scenario):
    pairs, structural, pos, st = scenario
    _, tier, _ = _stubbed().predict(structural, pos, st, pairs)
    assert (tier >= 0).all()                 # -1 sentinel must never survive


def test_only_undecided_rows_are_passed_downstream(scenario):
    """Each tier must be called with exactly the rows still unresolved — that is
    the entire point of the cost-aware cascade."""
    pairs, structural, pos, st = scenario
    m = _stubbed()
    m.predict(structural, pos, st, pairs)

    # tier 1 sees all five non-self-loop rows, by original index
    assert m.tier1.calls[0][1] == [101, 102, 103, 104, 105]
    # tier 2 sees only rows 103, 104, 105 (p1 = .50 .60 .75)
    assert m.tier2.calls[0][1] == pytest.approx([0.90, 0.50, 0.35])
    # tier 3 sees only rows 104, 105
    assert m.tier3.calls[0][1] == pytest.approx([0.30, 0.80])


def test_downstream_tiers_are_not_called_when_tier_one_resolves_everything():
    pairs = pd.DataFrame({'id1': [1, 1], 'id2': [2, 3]})
    m = _stubbed()
    m.predict(pd.DataFrame({'p': [0.99, 0.01]}), np.zeros((2, 3)), np.zeros(2), pairs)
    assert len(m.tier1.calls) == 1
    assert m.tier2.calls == []
    assert m.tier3.calls == []


def test_predictions_and_scores_agree_at_the_resolving_tier(scenario):
    pairs, structural, pos, st = scenario
    pred, tier, scores = _stubbed().predict(structural, pos, st, pairs)
    non_tier0 = tier != 0
    assert ((scores[non_tier0] >= 0.5) == (pred[non_tier0] == 1)).all()


def test_row_order_preserved_under_shuffled_index():
    pairs = pd.DataFrame({'id1': [1, 7, 1], 'id2': [2, 7, 3]}, index=['z', 'y', 'x'])
    structural = pd.DataFrame({'p': [0.99, 0.5, 0.5]}, index=pairs.index)
    pos = np.zeros((3, 2))
    pos[:, 0] = [0.0, 0.0, 0.95]
    st = np.array([0.0, 0.0, 0.0])
    _, tier, _ = _stubbed().predict(structural, pos, st, pairs)
    assert tier.tolist() == [1, 0, 2]


# ── thresholds ───────────────────────────────────────────────────────────────

def test_default_thresholds():
    m = CascadeLP()
    assert m.tier1_threshold == 0.8
    assert m.tier2_threshold == 0.7


def test_raising_tier1_threshold_never_increases_tier1_usage():
    rng = np.random.default_rng(11)
    n = 400
    pairs = pd.DataFrame({'id1': np.arange(n), 'id2': np.arange(n) + 10_000})
    structural = pd.DataFrame({'p': rng.uniform(0, 1, n)})
    pos = np.zeros((n, 2))
    pos[:, 0] = rng.uniform(0, 1, n)
    st = rng.uniform(0, 1, n)

    counts = []
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.01]:
        _, tier, _ = _stubbed(tier1_threshold=thr).predict(structural, pos, st, pairs)
        counts.append(int((tier == 1).sum()))

    assert counts == sorted(counts, reverse=True)
    assert counts[0] > 0
    assert counts[-1] == 0                   # threshold above 1.0 → tier 1 unusable


def test_raising_tier2_threshold_never_decreases_tier3_usage():
    rng = np.random.default_rng(12)
    n = 400
    pairs = pd.DataFrame({'id1': np.arange(n), 'id2': np.arange(n) + 10_000})
    structural = pd.DataFrame({'p': np.full(n, 0.5)})   # nothing resolves at tier 1
    pos = np.zeros((n, 2))
    pos[:, 0] = rng.uniform(0, 1, n)
    st = rng.uniform(0, 1, n)

    counts = []
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        _, tier, _ = _stubbed(tier2_threshold=thr).predict(structural, pos, st, pairs)
        assert (tier == 1).sum() == 0
        counts.append(int((tier == 3).sum()))

    assert counts == sorted(counts)


def test_threshold_of_zero_resolves_everything_at_tier_one():
    n = 20
    pairs = pd.DataFrame({'id1': np.arange(n), 'id2': np.arange(n) + 100})
    m = _stubbed(tier1_threshold=0.0)
    _, tier, _ = m.predict(pd.DataFrame({'p': np.full(n, 0.5)}),
                           np.zeros((n, 2)), np.zeros(n), pairs)
    assert (tier == 1).all()
    assert m.tier2.calls == []


# ── tier_stats ───────────────────────────────────────────────────────────────

def test_tier_stats_counts_and_percentages_sum_to_the_input(scenario):
    pairs, structural, pos, st = scenario
    m = _stubbed()
    _, tier, _ = m.predict(structural, pos, st, pairs)
    stats = m.tier_stats(tier)

    assert set(stats) == {'tier0', 'tier1', 'tier2', 'tier3'}
    assert sum(s['n'] for s in stats.values()) == len(pairs)
    assert sum(s['pct'] for s in stats.values()) == pytest.approx(100.0)
    assert stats['tier0']['n'] == 1
    assert stats['tier1']['n'] == 2
    assert stats['tier2']['n'] == 1
    assert stats['tier3']['n'] == 2
    assert stats['tier1']['pct'] == pytest.approx(200 / 6)
    assert all(isinstance(s['n'], int) for s in stats.values())


def test_tier_stats_reports_zero_for_unused_tiers():
    m = _stubbed()
    stats = m.tier_stats(np.array([3, 3, 3]))
    assert stats['tier0']['n'] == 0 and stats['tier0']['pct'] == 0.0
    assert stats['tier3']['pct'] == pytest.approx(100.0)


# ── fit ──────────────────────────────────────────────────────────────────────

def test_fit_excludes_self_loops_from_every_tier():
    pairs = pd.DataFrame({'id1': [1, 7, 2, 8], 'id2': [2, 7, 3, 8]})
    structural = pd.DataFrame({'p': [0.1, 0.2, 0.3, 0.4]})
    pos = np.zeros((4, 3))
    st = np.zeros(4)
    y = np.array([0, 1, 1, 1])

    m = _stubbed()
    assert m.fit(structural, pos, st, y, pairs) is m
    assert m.tier1.calls[0] == ('fit', 2, None)
    assert m.tier2.calls[0] == ('fit', 2)
    assert m.tier3.calls[0] == ('fit', 2)


def test_fit_masks_the_node2vec_block_too():
    pairs = pd.DataFrame({'id1': [1, 7, 2], 'id2': [2, 7, 3]})
    m = _stubbed()
    m.fit(pd.DataFrame({'p': [0.1, 0.2, 0.3]}), np.zeros((3, 3)), np.zeros(3),
          np.array([0, 1, 1]), pairs, n2v=np.zeros((3, 4)))
    assert m.tier1.calls[0] == ('fit', 2, 2)


def test_fit_all_self_loops_yields_empty_training_sets():
    pairs = pd.DataFrame({'id1': [7, 8], 'id2': [7, 8]})
    m = _stubbed()
    m.fit(pd.DataFrame({'p': [0.1, 0.2]}), np.zeros((2, 3)), np.zeros(2),
          np.array([1, 1]), pairs)
    assert m.tier1.calls[0] == ('fit', 0, None)


# ── save / load ──────────────────────────────────────────────────────────────

def test_save_load_round_trip_preserves_routing(tmp_path, scenario):
    pairs, structural, pos, st = scenario
    m = _stubbed()
    expected = m.predict(structural, pos, st, pairs)

    path = tmp_path / 'cascade.joblib'
    m.save(str(path))
    loaded = CascadeLP.load(str(path))
    assert isinstance(loaded, CascadeLP)
    assert loaded.tier1_threshold == m.tier1_threshold
    assert loaded.tier2_threshold == m.tier2_threshold

    got = loaded.predict(structural, pos, st, pairs)
    assert got[0].tolist() == expected[0].tolist()
    assert got[1].tolist() == expected[1].tolist()
    assert got[2].tolist() == pytest.approx(expected[2].tolist())


# ── end-to-end with the real tier classifiers ────────────────────────────────

def test_end_to_end_with_real_tiers_is_consistent():
    """No stubs: fit the actual Structural/POS/Embedding tiers on small separable
    synthetic data and check the cascade produces valid, self-consistent output."""
    rng = np.random.default_rng(3)
    n = 60
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    cn = np.concatenate([rng.integers(0, 2, n), rng.integers(8, 12, n)]).astype(float)
    structural = pd.DataFrame({
        'cn': cn, 'jaccard': cn / 20, 'adamic_adar': cn * 0.9, 'pref_attach': cn * 3,
        'p': cn,   # ignored by the real StructuralClassifier
    })
    pos = np.repeat(y.reshape(-1, 1).astype(float), 72, axis=1) + rng.normal(0, 0.05, (2 * n, 72))
    st = y * 0.8 + rng.normal(0, 0.05, 2 * n)

    pairs = pd.DataFrame({'id1': np.arange(2 * n), 'id2': np.arange(2 * n) + 10_000})
    pairs.iloc[0, 1] = pairs.iloc[0, 0]      # make row 0 a self-loop

    m = CascadeLP()
    m.tier2 = type(m.tier2)(n_estimators=25)
    m.fit(structural, pos, st, y, pairs)

    pred, tier, scores = m.predict(structural, pos, st, pairs)
    assert pred.shape == tier.shape == scores.shape == (2 * n,)
    assert set(np.unique(pred)) <= {0, 1}
    assert (tier >= 0).all() and (tier <= 3).all()
    assert tier[0] == 0 and pred[0] == 1 and scores[0] == 1.0
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    assert sum(s['n'] for s in m.tier_stats(tier).values()) == 2 * n
    assert (pred[1:] == y[1:]).mean() > 0.9


# ── documented-but-unimplemented behaviour ───────────────────────────────────

@pytest.mark.xfail(
    strict=True,
    reason='BUG/DOC MISMATCH: CascadeLP docstring and CLAUDE.md both state that '
           'cold-start pairs "skip straight to Tier 2", but predict() has no '
           'cold-start notion at all — every non-self-loop pair goes through '
           'Tier 1, and a confident structural prediction on all-zero features '
           'terminates there. See src/models/cascade.py:14 and :92-96.',
)
def test_cold_start_pairs_skip_tier_one():
    # A cold-start pair: neither node is in the graph, so every structural
    # heuristic is 0 and the structural model has nothing to work with.
    pairs = pd.DataFrame({'id1': [1], 'id2': [2]})
    structural = pd.DataFrame({'p': [0.95], 'cn': [0.0], 'jaccard': [0.0],
                               'adamic_adar': [0.0], 'pref_attach': [0.0]})
    m = _stubbed()
    _, tier, _ = m.predict(structural, np.full((1, 3), 0.9), np.zeros(1), pairs)
    assert tier.tolist() == [2]
