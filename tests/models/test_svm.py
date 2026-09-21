"""Tests for the five baseline classifiers in src/models/svm.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.svm import (
    EmbeddingClassifier,
    PosClassifier,
    StructuralClassifier,
    SvmClassifier,
    TfidfClassifier,
)

SCORE_MODELS = [TfidfClassifier, EmbeddingClassifier, SvmClassifier]


def _assert_proba_matrix(proba, n):
    assert proba.shape == (n, 2)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    assert proba.sum(axis=1).tolist() == pytest.approx([1.0] * n)


# ── StructuralClassifier ─────────────────────────────────────────────────────

def test_structural_feature_list_is_the_four_heuristics():
    assert StructuralClassifier.FEATURES == ['cn', 'jaccard', 'adamic_adar', 'pref_attach']


def test_structural_fit_predict_on_separable_data(separable_structural):
    X, y = separable_structural
    clf = StructuralClassifier().fit(X, y)
    proba = clf.predict_proba(X)
    _assert_proba_matrix(proba, len(y))
    pred = clf.predict(X)
    assert pred.tolist() == proba.argmax(axis=1).tolist()
    assert (pred == y).mean() > 0.95


def test_structural_ignores_extra_columns(separable_structural):
    X, y = separable_structural
    noisy = X.assign(irrelevant=np.arange(len(X)) * 1000.0)
    clf = StructuralClassifier().fit(noisy, y)
    a = clf.predict_proba(noisy)
    b = clf.predict_proba(X.assign(irrelevant=0.0))
    np.testing.assert_allclose(a, b)


def test_structural_fills_nan_features_with_zero(separable_structural):
    X, y = separable_structural
    clf = StructuralClassifier().fit(X, y)
    with_nan = X.copy()
    with_nan.iloc[0] = np.nan
    proba = clf.predict_proba(with_nan)
    assert np.isfinite(proba).all()

    zeroed = X.copy()
    zeroed.iloc[0] = 0.0
    np.testing.assert_allclose(proba[0], clf.predict_proba(zeroed)[0])



def test_structural_save_load_round_trip(tmp_path, separable_structural):
    X, y = separable_structural
    clf = StructuralClassifier().fit(X, y)
    path = tmp_path / 'structural.joblib'
    clf.save(str(path))
    loaded = StructuralClassifier.load(str(path))
    assert isinstance(loaded, StructuralClassifier)
    np.testing.assert_allclose(loaded.predict_proba(X), clf.predict_proba(X))


def test_structural_predict_proba_before_fit_raises(separable_structural):
    X, _ = separable_structural
    with pytest.raises(Exception):
        StructuralClassifier().predict_proba(X)


# ── score-based classifiers (TF-IDF / Embedding / SVM) ───────────────────────

@pytest.mark.parametrize('cls', SCORE_MODELS)
def test_score_models_fit_predict(cls, separable_scores):
    scores, y = separable_scores
    clf = cls().fit(scores, y)
    proba = clf.predict_proba(scores)
    _assert_proba_matrix(proba, len(y))
    assert (clf.predict(scores) == y).mean() > 0.95


@pytest.mark.parametrize('cls', SCORE_MODELS)
def test_score_models_accept_1d_input_of_any_length(cls, separable_scores):
    scores, y = separable_scores
    clf = cls().fit(scores, y)
    assert clf.predict_proba(np.array([0.1])).shape == (1, 2)
    assert clf.predict_proba(scores[:5]).shape == (5, 2)


@pytest.mark.parametrize('cls', [TfidfClassifier, EmbeddingClassifier])
def test_score_models_are_monotonic_in_the_score(cls, separable_scores):
    """The logistic-regression score models sit on a single scalar feature, so
    P(y=1) must rise monotonically with it. (SvmClassifier is excluded: an RBF
    kernel with Platt scaling is not monotone in the input.)"""
    scores, y = separable_scores
    clf = cls().fit(scores, y)
    grid = np.linspace(0.0, 1.0, 11)
    p1 = clf.predict_proba(grid)[:, 1]
    assert np.all(np.diff(p1) >= -1e-9)


@pytest.mark.parametrize('cls', SCORE_MODELS + [StructuralClassifier, PosClassifier])
def test_all_models_save_load_round_trip(cls, tmp_path, separable_scores,
                                         separable_structural):
    scores, y = separable_scores
    if cls is StructuralClassifier:
        X, y = separable_structural
    elif cls is PosClassifier:
        X = np.repeat(scores.reshape(-1, 1), 72, axis=1)
    else:
        X = scores

    clf = cls().fit(X, y)
    path = tmp_path / f'{cls.__name__}.joblib'
    clf.save(str(path))
    loaded = cls.load(str(path))
    np.testing.assert_allclose(loaded.predict_proba(X), clf.predict_proba(X))
    assert loaded.predict(X).tolist() == clf.predict(X).tolist()


# ── PosClassifier ────────────────────────────────────────────────────────────

@pytest.fixture
def pos_data():
    rng = np.random.default_rng(2)
    n = 40
    neg = rng.normal(0.0, 0.05, (n, 72))
    pos = rng.normal(1.0, 0.05, (n, 72))
    X = np.vstack([neg, pos])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def test_pos_classifier_fit_predict_72_dim(pos_data):
    X, y = pos_data
    clf = PosClassifier(n_estimators=25).fit(X, y)
    _assert_proba_matrix(clf.predict_proba(X), len(y))
    assert (clf.predict(X) == y).all()


def test_pos_classifier_predict_matches_argmax_of_proba(pos_data):
    X, y = pos_data
    clf = PosClassifier(n_estimators=25).fit(X, y)
    # `predict` delegates to RandomForest.predict, not to argmax like the others
    assert clf.predict(X).tolist() == clf.predict_proba(X).argmax(axis=1).tolist()


def test_pos_classifier_is_deterministic(pos_data):
    X, y = pos_data
    a = PosClassifier(n_estimators=25, n_jobs=1).fit(X, y).predict_proba(X)
    b = PosClassifier(n_estimators=25, n_jobs=1).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)


def test_pos_classifier_feature_width_mismatch_raises(pos_data):
    X, y = pos_data
    clf = PosClassifier(n_estimators=10).fit(X, y)
    with pytest.raises(ValueError):
        clf.predict_proba(np.zeros((3, 36)))


# ── SvmClassifier subsampling ────────────────────────────────────────────────

def test_svm_subsample_returns_input_unchanged_when_small_enough():
    clf = SvmClassifier(subsample_size=100)
    X = np.arange(50).reshape(-1, 1).astype(float)
    y = np.array([0, 1] * 25)
    Xs, ys = clf._subsample(X, y)
    assert Xs is X and ys is y


def test_svm_subsample_caps_at_subsample_size_and_keeps_stratification():
    clf = SvmClassifier(subsample_size=100)
    n = 1000
    X = np.linspace(0, 1, n).reshape(-1, 1)
    y = np.array([0] * 700 + [1] * 300)
    Xs, ys = clf._subsample(X, y)
    assert len(ys) == 100
    assert Xs.shape == (100, 1)
    # 70/30 class balance preserved by the stratified split
    assert ys.mean() == pytest.approx(0.3, abs=0.02)


def test_svm_subsample_is_deterministic_for_a_given_random_state():
    a = SvmClassifier(subsample_size=50, random_state=7)
    b = SvmClassifier(subsample_size=50, random_state=7)
    c = SvmClassifier(subsample_size=50, random_state=8)
    X = np.linspace(0, 1, 400).reshape(-1, 1)
    y = np.array([0, 1] * 200)
    assert a._subsample(X, y)[0].tolist() == b._subsample(X, y)[0].tolist()
    assert a._subsample(X, y)[0].tolist() != c._subsample(X, y)[0].tolist()


def test_svm_default_subsample_size_is_20k():
    assert SvmClassifier().subsample_size == 20_000


def test_svm_fit_only_sees_the_subsample(separable_scores):
    """Support vectors cannot outnumber the training subsample."""
    scores, y = separable_scores
    big_scores = np.tile(scores, 10)
    big_y = np.tile(y, 10)
    clf = SvmClassifier(subsample_size=60).fit(big_scores, big_y)
    assert clf.clf.support_.shape[0] <= 60
    assert clf.predict_proba(big_scores).shape == (len(big_y), 2)


def test_svm_probability_estimates_enabled():
    assert SvmClassifier().clf.probability is True
    assert SvmClassifier().clf.kernel == 'rbf'
