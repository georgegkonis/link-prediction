"""Tests for src/features/structural.py.

Reference graph (see tests/conftest.py::TINY_EDGES):

    edges   (1,2) (1,3) (2,3) (2,4) (3,4) (4,5)
    degrees 1→2, 2→3, 3→3, 4→3, 5→1

Hand-computed heuristics:

    (1,4): N(1)={2,3} N(4)={2,3,5}
           CN=2, Jaccard=2/3, AA=2/ln3, PA=2*3=6
    (1,5): N(1)={2,3} N(5)={4}
           CN=0, Jaccard=0, AA=0, PA=2*1=2
    (2,5): N(2)={1,3,4} N(5)={4}
           CN=1, Jaccard=1/3, AA=1/ln3, PA=3*1=3
"""

import numpy as np
import pandas as pd
import pytest
from gensim.models import KeyedVectors

from src.features.structural import (
    compute_heuristics,
)


@pytest.fixture
def heuristic_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {'id1': [1, 1, 2, 1, 1], 'id2': [4, 5, 5, 1, 999]},
        index=[10, 11, 12, 13, 14],
    )


# ── compute_heuristics ───────────────────────────────────────────────────────

def test_compute_heuristics_exact_values(tiny_graph, heuristic_pairs):
    out = compute_heuristics(tiny_graph, heuristic_pairs, show_progress=False)

    assert list(out.columns) == ['cn', 'jaccard', 'adamic_adar', 'pref_attach']
    assert list(out.index) == [10, 11, 12, 13, 14]

    assert out.loc[10].tolist() == pytest.approx([2.0, 2 / 3, 2 / np.log(3), 6.0])
    assert out.loc[11].tolist() == pytest.approx([0.0, 0.0, 0.0, 2.0])
    assert out.loc[12].tolist() == pytest.approx([1.0, 1 / 3, 1 / np.log(3), 3.0])


def test_compute_heuristics_self_loop_is_nan(tiny_graph, heuristic_pairs):
    out = compute_heuristics(tiny_graph, heuristic_pairs, show_progress=False)
    assert out.loc[13].isna().all()          # pair (1, 1)


def test_compute_heuristics_missing_node_is_zero(tiny_graph, heuristic_pairs):
    out = compute_heuristics(tiny_graph, heuristic_pairs, show_progress=False)
    assert out.loc[14].tolist() == [0.0, 0.0, 0.0, 0.0]   # pair (1, 999)


def test_compute_heuristics_preserves_row_order_after_internal_partition(tiny_graph):
    """Rows are split into self-loop / valid / cold groups internally and
    re-concatenated; the result must be reindexed back to the input order."""
    pairs = pd.DataFrame(
        {'id1': [1, 7, 1, 8, 2], 'id2': [1, 7, 4, 999, 5]},
        index=['a', 'b', 'c', 'd', 'e'],
    )
    out = compute_heuristics(tiny_graph, pairs, show_progress=False)
    assert list(out.index) == ['a', 'b', 'c', 'd', 'e']
    assert out.loc['a'].isna().all()
    assert out.loc['b'].isna().all()
    assert out.loc['c', 'cn'] == 2.0
    assert out.loc['d', 'cn'] == 0.0
    assert out.loc['e', 'cn'] == 1.0


def test_compute_heuristics_all_self_loops(tiny_graph):
    """Regression: with no non-self-loop rows, `DataFrame.apply(axis=1)` returns
    an empty DataFrame instead of a boolean Series and the masking used to blow
    up with KeyError('id1')."""
    pairs = pd.DataFrame({'id1': [1, 2], 'id2': [1, 2]}, index=[5, 6])
    out = compute_heuristics(tiny_graph, pairs, show_progress=False)
    assert list(out.columns) == ['cn', 'jaccard', 'adamic_adar', 'pref_attach']
    assert list(out.index) == [5, 6]
    assert out.isna().all().all()


def test_compute_heuristics_empty_pairs(tiny_graph):
    out = compute_heuristics(tiny_graph, pd.DataFrame({'id1': [], 'id2': []}),
                             show_progress=False)
    assert len(out) == 0
    assert list(out.columns) == ['cn', 'jaccard', 'adamic_adar', 'pref_attach']


def test_compute_heuristics_reversed_argument_order_raises(tiny_graph, heuristic_pairs):
    """Documented footgun: the graph must come first. Reversing the arguments
    must fail loudly rather than return garbage."""
    with pytest.raises(KeyError):
        compute_heuristics(heuristic_pairs, tiny_graph, show_progress=False)


def test_compute_heuristics_is_symmetric(tiny_graph):
    fwd = compute_heuristics(tiny_graph, pd.DataFrame({'id1': [1], 'id2': [4]}),
                             show_progress=False)
    rev = compute_heuristics(tiny_graph, pd.DataFrame({'id1': [4], 'id2': [1]}),
                             show_progress=False)
