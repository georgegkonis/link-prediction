import numpy as np
import pandas as pd
import pytest

from src.utils.difficulty import (
    DIFFICULTY_LEVELS,
    label_difficulty,
    pick_thresholds,
    trivial_mask,
)


# ── pick_thresholds ──────────────────────────────────────────────────────────

def test_pick_thresholds_uses_negative_class_percentile_only():
    # negatives: cn = 0..100, tfidf = 0.00..1.00 (101 evenly spaced values)
    # positives carry extreme values that must be ignored entirely
    y = np.concatenate([np.zeros(101, dtype=int), np.ones(5, dtype=int)])
    cn = np.concatenate([np.arange(101, dtype=float), np.full(5, 1e6)])
    tfidf = np.concatenate([np.linspace(0, 1, 101), np.full(5, 9.0)])

    cn_thr, tfidf_thr = pick_thresholds(y, cn, tfidf, fpr=0.5)
    assert cn_thr == pytest.approx(50.0)        # 50th percentile of 0..100
    assert tfidf_thr == pytest.approx(0.5)

    cn_thr, tfidf_thr = pick_thresholds(y, cn, tfidf, fpr=0.01)
    assert cn_thr == pytest.approx(99.0)        # 99th percentile of 0..100
    assert tfidf_thr == pytest.approx(0.99)


def test_pick_thresholds_default_fpr_is_one_percent():
    y = np.zeros(101, dtype=int)
    cn = np.arange(101, dtype=float)
    tfidf = np.arange(101, dtype=float)
    assert pick_thresholds(y, cn, tfidf) == pick_thresholds(y, cn, tfidf, fpr=0.01)


def test_pick_thresholds_ignores_nans():
    y = np.zeros(5, dtype=int)
    cn = np.array([0.0, 1.0, 2.0, np.nan, np.nan])
    tfidf = np.array([np.nan, 0.0, 0.5, 1.0, np.nan])
    cn_thr, tfidf_thr = pick_thresholds(y, cn, tfidf, fpr=0.5)
    assert cn_thr == pytest.approx(1.0)     # median of [0, 1, 2]
    assert tfidf_thr == pytest.approx(0.5)  # median of [0, 0.5, 1]
    assert not np.isnan(cn_thr)


def test_pick_thresholds_returns_python_floats():
    y = np.zeros(3, dtype=int)
    a = np.array([1.0, 2.0, 3.0])
    cn_thr, tfidf_thr = pick_thresholds(y, a, a)
    assert type(cn_thr) is float and type(tfidf_thr) is float


# ── label_difficulty ─────────────────────────────────────────────────────────

@pytest.fixture
def labelled():
    pairs = pd.DataFrame(
        {
            'id1': [7, 1, 3, 5, 8, 10],
            'id2': [7, 2, 4, 6, 9, 11],
        },
        index=[100, 101, 102, 103, 104, 105],
    )
    #        self-loop AND high cn AND high textsim  → self_loop wins
    #        high cn AND high textsim               → high_cn wins
    #        high textsim only                      → high_textsim
    #        neither                                → hard
    #        NaN features                           → hard
    #        exactly at both thresholds             → hard (strict >)
    cn = np.array([100.0, 100.0, 0.0, 0.0, np.nan, 1.0])
    tfidf = np.array([0.9, 0.9, 0.9, 0.0, np.nan, 0.5])
    return pairs, cn, tfidf


def test_label_difficulty_all_four_labels_and_precedence(labelled):
    pairs, cn, tfidf = labelled
    out = label_difficulty(pairs, cn, tfidf, cn_threshold=1.0, tfidf_threshold=0.5)
    assert out.tolist() == [
        'trivial_self_loop',
        'trivial_high_cn',
        'trivial_high_textsim',
        'hard',
        'hard',
        'hard',
    ]
    assert set(out) <= set(DIFFICULTY_LEVELS)


def test_label_difficulty_preserves_index_and_name(labelled):
    pairs, cn, tfidf = labelled
    out = label_difficulty(pairs, cn, tfidf, 1.0, 0.5)
    assert list(out.index) == [100, 101, 102, 103, 104, 105]
    assert out.name == 'difficulty'
    assert len(out) == len(pairs)


def test_label_difficulty_threshold_is_strictly_greater():
    pairs = pd.DataFrame({'id1': [1, 1], 'id2': [2, 2]})
    out = label_difficulty(pairs, np.array([5.0, 5.001]), np.array([0.0, 0.0]),
                           cn_threshold=5.0, tfidf_threshold=1.0)
    assert out.tolist() == ['hard', 'trivial_high_cn']


def test_label_difficulty_self_loop_wins_even_with_zero_features():
    pairs = pd.DataFrame({'id1': [4], 'id2': [4]})
    out = label_difficulty(pairs, np.array([0.0]), np.array([0.0]), 1.0, 0.5)
    assert out.iloc[0] == 'trivial_self_loop'


def test_trivial_mask(labelled):
    pairs, cn, tfidf = labelled
    out = label_difficulty(pairs, cn, tfidf, 1.0, 0.5)
    assert trivial_mask(out).tolist() == [True, True, True, False, False, False]


def test_pick_thresholds_feeds_label_difficulty_end_to_end():
    """With fpr=0.5 the CN threshold lands on the negative-class median, so
    roughly half the negatives must fall out as non-trivial on CN."""
    y = np.array([0, 0, 0, 0, 1, 1])
    cn = np.array([0.0, 1.0, 2.0, 3.0, 50.0, 60.0])
    tfidf = np.array([0.0, 0.1, 0.2, 0.3, 0.9, 0.9])
    cn_thr, tfidf_thr = pick_thresholds(y, cn, tfidf, fpr=0.5)
    assert cn_thr == pytest.approx(1.5)

    pairs = pd.DataFrame({'id1': [1, 2, 3, 4, 5, 6], 'id2': [11, 12, 13, 14, 15, 16]})
    out = label_difficulty(pairs, cn, tfidf, cn_thr, tfidf_thr)
    assert out.tolist()[:4] == ['hard', 'hard', 'trivial_high_cn', 'trivial_high_cn']
    assert out.tolist()[4:] == ['trivial_high_cn', 'trivial_high_cn']
