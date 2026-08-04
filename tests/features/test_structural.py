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
    _generate_walks,
    compute_heuristics,
    node2vec_hadamard_features,
    train_node2vec,
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
    np.testing.assert_allclose(fwd.values, rev.values)


# ── node2vec_hadamard_features ───────────────────────────────────────────────

@pytest.fixture
def tiny_kv() -> KeyedVectors:
    """A hand-built KeyedVectors — no training, no I/O."""
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(
        ['1', '2', '3'],
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, -1.0, 2.0]], dtype=np.float32),
    )
    return kv


def test_node2vec_hadamard_elementwise_product(tiny_kv):
    pairs = pd.DataFrame({'id1': [1, 1], 'id2': [2, 3]})
    out = node2vec_hadamard_features(tiny_kv, pairs, dim=3)
    assert out.shape == (2, 3)
    assert out.dtype == np.float32
    assert out[0].tolist() == pytest.approx([4.0, 10.0, 18.0])
    assert out[1].tolist() == pytest.approx([0.0, -2.0, 6.0])


def test_node2vec_hadamard_out_of_vocab_is_zero_vector(tiny_kv):
    pairs = pd.DataFrame({'id1': [1, 999, 999], 'id2': [999, 1, 998]})
    out = node2vec_hadamard_features(tiny_kv, pairs, dim=3)
    assert out.shape == (3, 3)
    assert (out == 0).all()


def test_node2vec_hadamard_mixed_vocab_keeps_row_alignment(tiny_kv):
    pairs = pd.DataFrame({'id1': [999, 1, 999], 'id2': [1, 2, 2]}, index=[5, 6, 7])
    out = node2vec_hadamard_features(tiny_kv, pairs, dim=3)
    assert (out[0] == 0).all()
    assert out[1].tolist() == pytest.approx([4.0, 10.0, 18.0])
    assert (out[2] == 0).all()


def test_node2vec_hadamard_keys_are_looked_up_as_strings(tiny_kv):
    """Node ids arrive as ints but the KeyedVectors keys are strings."""
    assert '1' in tiny_kv.key_to_index and 1 not in tiny_kv.key_to_index
    out = node2vec_hadamard_features(tiny_kv, pd.DataFrame({'id1': [1], 'id2': [1]}), dim=3)
    assert out[0].tolist() == pytest.approx([1.0, 4.0, 9.0])


def test_node2vec_hadamard_empty_pairs(tiny_kv):
    out = node2vec_hadamard_features(tiny_kv, pd.DataFrame({'id1': [], 'id2': []}), dim=3)
    assert out.shape == (0, 3)


def test_node2vec_hadamard_dim_mismatch_raises(tiny_kv):
    """`dim` is not validated against wv.vector_size — a mismatch must at least
    fail loudly instead of silently truncating."""
    with pytest.raises(ValueError):
        node2vec_hadamard_features(tiny_kv, pd.DataFrame({'id1': [1], 'id2': [2]}), dim=64)


# ── walk generation / training ───────────────────────────────────────────────

def test_generate_walks_shape_and_determinism(tiny_graph):
    walks = _generate_walks(tiny_graph, num_walks=2, walk_length=4, seed=7)
    assert len(walks) == 2 * tiny_graph.number_of_nodes()
    assert all(len(w) == 4 for w in walks)                  # no dead ends in this graph
    assert all(isinstance(tok, str) for w in walks for tok in w)
    assert walks == _generate_walks(tiny_graph, num_walks=2, walk_length=4, seed=7)


def test_generate_walks_are_valid_paths(tiny_graph):
    for walk in _generate_walks(tiny_graph, num_walks=1, walk_length=5, seed=3):
        ints = [int(t) for t in walk]
        for u, v in zip(ints, ints[1:]):
            assert tiny_graph.has_edge(u, v)


def test_generate_walks_stops_at_isolated_node():
    import networkx as nx
    G = nx.Graph()
    G.add_node(42)
    walks = _generate_walks(G, num_walks=1, walk_length=10, seed=0)
    assert walks == [['42']]


def test_train_node2vec_rejects_biased_walks(tiny_graph):
    with pytest.raises(NotImplementedError, match='unbiased DeepWalk'):
        train_node2vec(tiny_graph, p=0.5)
    with pytest.raises(NotImplementedError):
        train_node2vec(tiny_graph, q=2.0)


def test_train_node2vec_returns_keyed_vectors_for_every_node(tiny_graph):
    wv = train_node2vec(tiny_graph, dimensions=8, walk_length=5, num_walks=2,
                        workers=1, window=3, seed=42)
    assert set(wv.key_to_index) == {str(n) for n in tiny_graph.nodes()}
    assert wv.vectors.shape == (tiny_graph.number_of_nodes(), 8)

    # ...and those vectors feed straight into the Hadamard featuriser
    feats = node2vec_hadamard_features(
        wv, pd.DataFrame({'id1': [1], 'id2': [2]}), dim=8)
    assert feats.shape == (1, 8)
    assert not np.allclose(feats, 0)
