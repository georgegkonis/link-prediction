"""Tests for the pure audit helpers in scripts/analysis/audit_leakage.py."""

import pandas as pd
import pytest

from scripts.analysis.audit_leakage import (
    _undirected_key,
    intra_train_duplicates,
    pair_overlap,
    self_loop_report,
)


def test_undirected_key_is_order_invariant():
    df = pd.DataFrame({'id1': [1, 2, 5], 'id2': [2, 1, 5]})
    keys = _undirected_key(df)
    assert keys.tolist() == ['1_2', '1_2', '5_5']


def test_pair_overlap_counts_exact_and_reversed_separately():
    train = pd.DataFrame({'id1': [1, 3, 5], 'id2': [2, 4, 6]})
    test = pd.DataFrame({
        'id1': [1, 4, 9, 3],
        'id2': [2, 3, 9, 4],
    })
    out = pair_overlap(train, test)
    assert out['exact_count'] == 2                    # (1,2) and (3,4)
    assert out['exact_overlap'] == [(1, 2), (3, 4)]
    assert out['reversed_count'] == 1                 # (4,3) reverses train's (3,4)
    assert out['reversed_overlap'] == [(4, 3)]
    assert out['test_count'] == 4


def test_pair_overlap_reversed_excludes_pairs_already_counted_as_exact():
    """A symmetric pair present in train in both orientations must not be double
    counted."""
    train = pd.DataFrame({'id1': [1, 2], 'id2': [2, 1]})
    test = pd.DataFrame({'id1': [1], 'id2': [2]})
    out = pair_overlap(train, test)
    assert out['exact_count'] == 1
    assert out['reversed_count'] == 0


def test_pair_overlap_no_overlap():
    train = pd.DataFrame({'id1': [1], 'id2': [2]})
    test = pd.DataFrame({'id1': [3], 'id2': [4]})
    out = pair_overlap(train, test)
    assert out['exact_count'] == 0 and out['reversed_count'] == 0


def test_pair_overlap_counts_duplicated_test_rows_once_each():
    train = pd.DataFrame({'id1': [1], 'id2': [2]})
    test = pd.DataFrame({'id1': [1, 1], 'id2': [2, 2]})
    assert pair_overlap(train, test)['exact_count'] == 2


def test_intra_train_duplicates_finds_undirected_repeats():
    train = pd.DataFrame({
        'id1': [1, 2, 5, 7],
        'id2': [2, 1, 6, 8],
        'label': [1, 1, 0, 1],
    })
    dupes = intra_train_duplicates(train)
    assert len(dupes) == 2                            # (1,2) and (2,1)
    assert dupes['_key'].unique().tolist() == ['1_2']
    assert not dupes['label_conflict'].any()


def test_intra_train_duplicates_flags_label_conflicts():
    train = pd.DataFrame({
        'id1': [1, 2, 3, 4],
        'id2': [2, 1, 4, 3],
        'label': [1, 0, 1, 1],
    })
    dupes = intra_train_duplicates(train)
    conflict_by_key = dupes.groupby('_key')['label_conflict'].first()
    assert conflict_by_key['1_2'] is True or conflict_by_key['1_2']
    assert not conflict_by_key['3_4']


def test_intra_train_duplicates_empty_when_all_unique():
    train = pd.DataFrame({'id1': [1, 3], 'id2': [2, 4], 'label': [1, 0]})
    assert intra_train_duplicates(train).empty


def test_self_loop_report_with_labels():
    df = pd.DataFrame({
        'id1': [1, 7, 8, 8],
        'id2': [2, 7, 8, 8],
        'label': [1, 1, 1, 0],
    })
    rep = self_loop_report(df, 'train')
    assert rep['name'] == 'train'
    assert rep['total'] == 4
    assert rep['self_loops'] == 3
    assert rep['label_counts'] == {1: 2, 0: 1}


def test_self_loop_report_without_labels_omits_counts():
    df = pd.DataFrame({'id1': [1, 7], 'id2': [2, 7]})
    rep = self_loop_report(df, 'test')
    assert rep['self_loops'] == 1
    assert 'label_counts' not in rep


def test_self_loop_report_none_present():
    df = pd.DataFrame({'id1': [1], 'id2': [2], 'label': [1]})
    rep = self_loop_report(df, 'x')
    assert rep['self_loops'] == 0
    assert rep['label_counts'] == {}
