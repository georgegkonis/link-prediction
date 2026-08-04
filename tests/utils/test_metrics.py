"""Exact-value tests for src/utils/metrics.py.

The reference confusion matrix used throughout:

    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 0, 1]

    class 1: TP=2 FP=1 FN=1 → P=2/3  R=2/3  F1=2/3
    class 0: TP=1 FP=1 FN=1 → P=1/2  R=1/2  F1=1/2
    macro F1 = (2/3 + 1/2) / 2 = 0.5833333...
"""

import time

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import (
    EvalResult,
    cold_start_mask,
    evaluate,
    evaluate_by_group,
    tier_difficulty_breakdown,
    timer,
)

Y_TRUE = np.array([1, 1, 1, 0, 0])
Y_PRED = np.array([1, 1, 0, 0, 1])
MACRO_F1 = (2 / 3 + 1 / 2) / 2


# ── cold_start_mask ──────────────────────────────────────────────────────────

def test_cold_start_mask_hand_computed(tiny_graph):
    pairs = pd.DataFrame(
        {'id1': [1, 1, 2, 1, 1, 999], 'id2': [4, 5, 5, 1, 999, 998]},
        index=[10, 11, 12, 13, 14, 15],
    )
    mask = cold_start_mask(pairs, tiny_graph)
    # (1,4): common nbrs {2,3}  → warm
    # (1,5): N(1)={2,3}, N(5)={4} → no common nbr → cold
    # (2,5): common nbr {4}     → warm
    # (1,1): self-loop          → never cold by construction
    # (1,999)/(999,998): node absent from graph → cold
    assert mask.tolist() == [False, True, False, False, True, True]
    assert mask.dtype == bool


def test_cold_start_mask_empty_graph():
    G = nx.Graph()
    pairs = pd.DataFrame({'id1': [1, 2], 'id2': [3, 4]})
    assert cold_start_mask(pairs, G).tolist() == [True, True]


# ── evaluate ─────────────────────────────────────────────────────────────────

def test_evaluate_macro_f1_exact():
    res = evaluate(Y_TRUE, Y_PRED)
    assert res.macro_f1 == pytest.approx(MACRO_F1)
    assert np.isnan(res.auc_roc)          # no scores supplied
    assert res.cold_start_macro_f1 is None
    assert res.cold_start_count == 0
    assert res.latency_ms is None


def test_evaluate_auc_exact():
    # positives score .9 .8 .4 ; negatives score .3 .6
    # concordant pairs: 5 of 6 → AUC = 5/6
    scores = np.array([0.9, 0.8, 0.4, 0.3, 0.6])
    res = evaluate(Y_TRUE, Y_PRED, y_scores=scores)
    assert res.auc_roc == pytest.approx(5 / 6)


def test_evaluate_cold_start_subset_exact():
    cs = np.array([True, False, True, False, True])
    # subset: y_true=[1,1,0], y_pred=[1,0,1]
    #   class 1: TP=1 FP=1 FN=1 → F1=1/2
    #   class 0: TP=0 FP=1 FN=1 → F1=0
    res = evaluate(Y_TRUE, Y_PRED, cs_mask=cs)
    assert res.cold_start_count == 3
    assert res.cold_start_macro_f1 == pytest.approx(0.25)


def test_evaluate_empty_cold_start_mask_is_ignored():
    res = evaluate(Y_TRUE, Y_PRED, cs_mask=np.zeros(5, dtype=bool))
    assert res.cold_start_macro_f1 is None
    assert res.cold_start_count == 0


def test_evaluate_latency_passthrough():
    assert evaluate(Y_TRUE, Y_PRED, latency_ms=12.5).latency_ms == 12.5


def test_evaluate_perfect_and_inverted_predictions():
    assert evaluate(Y_TRUE, Y_TRUE).macro_f1 == pytest.approx(1.0)
    assert evaluate(Y_TRUE, 1 - Y_TRUE).macro_f1 == pytest.approx(0.0)


def test_evaluate_single_class_y_true_gives_nan_auc_with_a_warning():
    """AUC is undefined for a single-class y_true. sklearn warns and returns NaN;
    `evaluate` passes that NaN straight through, so any downstream aggregation
    (means, comparisons, JSON dumps) has to cope with it."""
    y = np.ones(4, dtype=int)
    with pytest.warns(UserWarning, match='Only one class'):
        res = evaluate(y, y, y_scores=np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.isnan(res.auc_roc)


def test_evaluate_single_class_without_scores_is_fine():
    y = np.ones(4, dtype=int)
    res = evaluate(y, y)
    # macro over the two label slots sklearn sees (only class 1 present) → 1.0
    assert res.macro_f1 == pytest.approx(1.0)


def test_eval_result_str_includes_optional_fields():
    plain = str(EvalResult(macro_f1=0.5, auc_roc=0.6))
    assert 'Macro F1' in plain and 'AUC-ROC' in plain
    assert 'Cold-start' not in plain and 'Latency' not in plain

    full = str(EvalResult(macro_f1=0.5, auc_roc=0.6,
                          cold_start_macro_f1=0.4, cold_start_count=7,
                          latency_ms=3.0))
    assert 'Cold-start F1     : 0.4000  (n=7)' in full
    assert 'Latency           : 3.0 ms' in full


def test_eval_result_extra_defaults_to_independent_dict():
    a, b = EvalResult(0.1, 0.2), EvalResult(0.1, 0.2)
    a.extra['k'] = 1
    assert b.extra == {}


# ── evaluate_by_group ────────────────────────────────────────────────────────

def test_evaluate_by_group_exact():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    group = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    df = evaluate_by_group(y_true, y_pred, group)

    assert list(df.index) == ['a', 'b']
    assert df.loc['a', 'n'] == 3
    assert df.loc['a', 'accuracy'] == pytest.approx(1.0)
    assert df.loc['a', 'macro_f1'] == pytest.approx(1.0)

    # group b: y_true=[0,1,0], y_pred=[1,0,0] → 1/3 correct
    #   class 0: TP=1 FP=1 FN=1 → F1=1/2 ; class 1: TP=0 → F1=0
    assert df.loc['b', 'n'] == 3
    assert df.loc['b', 'accuracy'] == pytest.approx(1 / 3)
    assert df.loc['b', 'macro_f1'] == pytest.approx(0.25)


def test_evaluate_by_group_single_class_group_uses_zero_division_guard():
    y_true = np.array([1, 1])
    y_pred = np.array([1, 0])
    group = np.array(['g', 'g'])
    df = evaluate_by_group(y_true, y_pred, group)
    # class 1: TP=1 FP=0 FN=1 → F1=2/3 ; class 0 absent → 0 (zero_division=0)
    assert df.loc['g', 'macro_f1'] == pytest.approx((2 / 3) / 2)


def test_evaluate_by_group_empty_input_returns_empty_frame():
    df = evaluate_by_group(np.array([]), np.array([]), np.array([]))
    assert df.empty


def test_evaluate_by_group_preserves_first_seen_group_order():
    group = np.array(['z', 'a', 'z', 'a'])
    df = evaluate_by_group(np.ones(4, dtype=int), np.ones(4, dtype=int), group)
    assert list(df.index) == ['z', 'a']


# ── tier_difficulty_breakdown ────────────────────────────────────────────────

def test_tier_difficulty_breakdown_exact():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    tier = np.array([1, 1, 1, 3, 3, 3])
    difficulty = pd.Series(
        ['trivial_high_cn', 'trivial_high_cn', 'trivial_high_cn', 'hard', 'hard', 'hard'],
        index=[100, 101, 102, 103, 104, 105],
    )
    df = tier_difficulty_breakdown(y_true, y_pred, tier, difficulty)

    assert set(df.index) == {(1, 'trivial_high_cn'), (3, 'hard')}
    assert df.loc[[(1, 'trivial_high_cn')], 'n'].iloc[0] == 3
    # tier 1: y_true=[1,1,0], y_pred=[1,0,0] → 2/3 accuracy
    assert df.loc[[(1, 'trivial_high_cn')], 'accuracy'].iloc[0] == pytest.approx(2 / 3)
    # tier 3: y_true=[0,1,0], y_pred=[0,1,1] → 2/3 accuracy
    assert df.loc[[(3, 'hard')], 'accuracy'].iloc[0] == pytest.approx(2 / 3)


def test_tier_difficulty_breakdown_ignores_difficulty_index():
    """`difficulty.values` is used positionally, so a mismatched index must not
    reorder or realign the cross-tab."""
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0])
    tier = np.array([0, 1])
    diff = pd.Series(['trivial_self_loop', 'hard'], index=[999, 5])
    df = tier_difficulty_breakdown(y_true, y_pred, tier, diff)
    assert set(df.index) == {(0, 'trivial_self_loop'), (1, 'hard')}
    assert df['accuracy'].tolist() == [1.0, 1.0]


def test_tier_difficulty_breakdown_empty():
    df = tier_difficulty_breakdown(
        np.array([]), np.array([]), np.array([]), pd.Series([], dtype=object))
    assert df.empty


# ── timer ────────────────────────────────────────────────────────────────────

def test_timer_records_elapsed_ms_only_after_block():
    with timer() as t:
        assert t == []          # nothing recorded until the block exits
        time.sleep(0.01)
    assert len(t) == 1
    assert t[0] >= 10.0         # at least the 10 ms we slept
    assert t[0] < 5000.0


def test_timer_does_not_swallow_exceptions():
    with pytest.raises(RuntimeError):
        with timer():
            raise RuntimeError('boom')
