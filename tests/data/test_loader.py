"""Tests for src/data/loader.py. All files are synthesised into tmp_path."""

import networkx as nx
import pandas as pd
import pytest

from src.data.loader import build_graph, load_edges, load_nodes, load_nodes_for_ids


# ── load_edges ───────────────────────────────────────────────────────────────

@pytest.fixture
def edges_csv(tmp_path):
    p = tmp_path / 'train.csv'
    p.write_text(
        'id,id1,id2,label\n'
        '0,1,2,1\n'
        '1,1,3,1\n'
        '2,2,3,0\n'
        '3,7,7,1\n'
    )
    return p


def test_load_edges_uses_id_as_index(edges_csv):
    df = load_edges(str(edges_csv))
    assert df.index.name == 'id'
    assert list(df.columns) == ['id1', 'id2', 'label']
    assert len(df) == 4
    assert df.loc[2].tolist() == [2, 3, 0]


def test_load_edges_nrows_truncates(edges_csv):
    assert len(load_edges(str(edges_csv), nrows=2)) == 2


def test_load_edges_missing_id_column_raises(tmp_path):
    p = tmp_path / 'bad.csv'
    p.write_text('id1,id2,label\n1,2,1\n')
    with pytest.raises(ValueError):
        load_edges(str(p))


# ── load_nodes / load_nodes_for_ids ──────────────────────────────────────────

@pytest.fixture
def nodes_tsv(tmp_path):
    """10 nodes, ids 0..9, so chunk boundaries can be exercised precisely."""
    lines = ['id\ttext'] + [f'{i}\ttext for node {i}' for i in range(10)]
    p = tmp_path / 'nodes.tsv'
    p.write_text('\n'.join(lines) + '\n')
    return p


def test_load_nodes_reads_tab_separated(nodes_tsv):
    df = load_nodes(str(nodes_tsv))
    assert df.index.name == 'id'
    assert len(df) == 10
    assert df.loc[3, 'text'] == 'text for node 3'


def test_load_nodes_nrows(nodes_tsv):
    assert len(load_nodes(str(nodes_tsv), nrows=4)) == 4


def test_load_nodes_for_ids_returns_only_requested(nodes_tsv):
    df = load_nodes_for_ids(str(nodes_tsv), {2, 5, 9}, chunksize=3)
    assert sorted(df.index) == [2, 5, 9]
    assert df.loc[5, 'text'] == 'text for node 5'
    assert list(df.columns) == ['text']


@pytest.mark.parametrize('chunksize', [1, 2, 3, 4, 5, 9, 10, 11, 50_000])
def test_load_nodes_for_ids_is_chunksize_invariant(nodes_tsv, chunksize):
    """Chunk boundaries must not drop or duplicate rows — request ids that
    straddle every plausible boundary."""
    wanted = {0, 2, 3, 6, 9}
    df = load_nodes_for_ids(str(nodes_tsv), wanted, chunksize=chunksize)
    assert sorted(df.index) == sorted(wanted)
    assert not df.index.duplicated().any()


def test_load_nodes_for_ids_ignores_unknown_ids(nodes_tsv):
    df = load_nodes_for_ids(str(nodes_tsv), {5, 12345}, chunksize=4)
    assert sorted(df.index) == [5]


def test_load_nodes_for_ids_no_match_returns_empty_text_frame(nodes_tsv):
    df = load_nodes_for_ids(str(nodes_tsv), {999}, chunksize=4)
    assert df.empty
    assert list(df.columns) == ['text']


def test_load_nodes_for_ids_empty_request(nodes_tsv):
    df = load_nodes_for_ids(str(nodes_tsv), set(), chunksize=4)
    assert df.empty


def test_load_nodes_for_ids_preserves_file_order(nodes_tsv):
    df = load_nodes_for_ids(str(nodes_tsv), {9, 1, 5}, chunksize=2)
    assert list(df.index) == [1, 5, 9]


def test_load_nodes_for_ids_output_indexes_by_id(nodes_tsv):
    """Downstream code does `nodes.loc[node_id, 'text']`, so the index must be
    the node id, not a positional range."""
    df = load_nodes_for_ids(str(nodes_tsv), {7}, chunksize=3)
    assert df.loc[7, 'text'] == 'text for node 7'


# ── build_graph ──────────────────────────────────────────────────────────────

def test_build_graph_uses_positive_edges_only(tiny_edges_df):
    G = build_graph(tiny_edges_df)
    assert isinstance(G, nx.Graph)
    for u, v in [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)]:
        assert G.has_edge(u, v)
    assert not G.has_edge(1, 5)      # label 0
    assert not G.has_edge(3, 5)      # label 0


def test_build_graph_is_undirected(tiny_edges_df):
    G = build_graph(tiny_edges_df)
    assert G.has_edge(2, 1) and G.has_edge(1, 2)


def test_build_graph_keeps_positive_self_loops(tiny_edges_df):
    """Node 7 only appears as a positive self-loop; it lands in the graph as a
    degree-1 self-loop node. Callers filter self-loops separately (Tier 0)."""
    G = build_graph(tiny_edges_df)
    assert G.has_node(7)
    assert G.has_edge(7, 7)
    assert list(nx.selfloop_edges(G)) == [(7, 7)]


def test_build_graph_excludes_nodes_that_only_have_negative_edges():
    edges = pd.DataFrame({'id1': [1, 10], 'id2': [2, 11], 'label': [1, 0]})
    G = build_graph(edges)
    assert set(G.nodes()) == {1, 2}


def test_build_graph_all_negative_gives_empty_graph():
    edges = pd.DataFrame({'id1': [1, 3], 'id2': [2, 4], 'label': [0, 0]})
    G = build_graph(edges)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_build_graph_deduplicates_repeated_edges():
    edges = pd.DataFrame({'id1': [1, 1, 2], 'id2': [2, 2, 1], 'label': [1, 1, 1]})
    G = build_graph(edges)
    assert G.number_of_edges() == 1


def test_build_graph_requires_label_column():
    with pytest.raises(KeyError):
        build_graph(pd.DataFrame({'id1': [1], 'id2': [2]}))


def test_loader_round_trip_edges_to_graph(edges_csv):
    G = build_graph(load_edges(str(edges_csv)))
    assert set(G.nodes()) == {1, 2, 3, 7}
    assert G.number_of_edges() == 3      # (1,2), (1,3), (7,7)
