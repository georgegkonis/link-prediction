"""Tests for scripts/train.py::_split — the train/val partitioning logic.

Only the pure helper is exercised; `main()` is a hydra entry point that reads
`data/interim/` and writes checkpoints.
"""

import math

import numpy as np
import pandas as pd
import pytest

from scripts.train import _split


@pytest.fixture
def data():
    n = 100
    pairs = pd.DataFrame(
        {'id1': np.arange(n), 'id2': np.arange(n) + 1000},
        index=pd.RangeIndex(n, name='id'),
    )
    # rows 0, 1, 2 are self-loops
    for i in range(3):
        pairs.iloc[i, 1] = pairs.iloc[i, 0]
    y = np.array([i % 2 for i in range(n)])
    return {
        'pairs': pairs,
        'y': y,
        'structural': pd.DataFrame({'cn': np.arange(n, dtype=float)}),
        'st_scores': np.arange(n, dtype=float),
    }


def test_split_excludes_self_loops_entirely(data):
    tr, val, _ = _split(data, 'cascade', val_size=0.2, seed=42)
    assert set(tr).isdisjoint(val)
    combined = set(tr) | set(val)
    assert combined == set(range(3, 100))          # rows 0-2 (self-loops) dropped
    assert len(combined) == 97


def test_split_respects_val_size(data):
    tr, val, _ = _split(data, 'cascade', val_size=0.2, seed=42)
    assert len(val) == math.ceil(97 * 0.2)      # sklearn rounds the val side up
    assert len(tr) + len(val) == 97


def test_split_is_stratified_on_the_label(data):
    tr, val, _ = _split(data, 'cascade', val_size=0.3, seed=0)
    y = data['y']
    assert y[tr].mean() == pytest.approx(y[val].mean(), abs=0.05)


def test_split_is_deterministic_for_a_seed(data):
    a = _split(data, 'cascade', 0.2, seed=7)[0]
    b = _split(data, 'cascade', 0.2, seed=7)[0]
    c = _split(data, 'cascade', 0.2, seed=8)[0]
    assert a.tolist() == b.tolist()
    assert a.tolist() != c.tolist()


def test_split_sub_helper_handles_frames_and_arrays(data):
    tr, val, sub = _split(data, 'cascade', 0.2, seed=42)

    s_tr, s_val = sub(data['structural'])
    assert isinstance(s_tr, pd.DataFrame)
    assert len(s_tr) == len(tr) and len(s_val) == len(val)
    assert s_tr['cn'].tolist() == data['structural']['cn'].values[tr].tolist()

    a_tr, a_val = sub(data['st_scores'])
    assert isinstance(a_tr, np.ndarray)
    assert a_tr.tolist() == data['st_scores'][tr].tolist()
    assert len(a_val) == len(val)


def test_split_sub_uses_positional_indexing_on_frames(data):
    """`sub` uses `.iloc`, so a non-positional pair index must not shift rows."""
    data = dict(data)
    data['structural'] = data['structural'].set_axis(
        pd.Index(np.arange(100) + 500, name='id'))
    tr, _, sub = _split(data, 'cascade', 0.2, seed=42)
    s_tr, _ = sub(data['structural'])
    assert s_tr['cn'].tolist() == data['structural']['cn'].values[tr].tolist()


def test_split_indices_are_positional_not_label_based(data):
    """Returned indices index into the *positional* rows of pairs/y."""
    tr, val, _ = _split(data, 'cascade', 0.2, seed=42)
    idx = np.concatenate([tr, val])
    sub_pairs = data['pairs'].iloc[idx]
    assert (sub_pairs['id1'] != sub_pairs['id2']).all()
