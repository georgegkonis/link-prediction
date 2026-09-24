import networkx as nx
import numpy as np
import pandas as pd

from scripts.wiki.build_dataset import build_sparse_benchmark


def _source_graph():
    graph = nx.gnp_random_graph(30, 0.18, seed=7)
    graph.add_edges_from((i, i + 1) for i in range(29))
    positives = pd.DataFrame(
        sorted((min(u, v), max(u, v)) for u, v in graph.edges()),
        columns=['id1', 'id2'],
    )
    return positives, np.arange(30)


def _pairs(frame):
    return set(frame[['id1', 'id2']].itertuples(index=False, name=None))


def test_sparse_benchmark_is_deterministic_and_uses_every_positive():
    positives, nodes = _source_graph()
    frames1, meta1 = build_sparse_benchmark(positives, nodes, edge_retention=0.8, seed=42)
    frames2, meta2 = build_sparse_benchmark(positives, nodes, edge_retention=0.8, seed=42)

    assert meta1 == meta2
    for name in frames1:
        pd.testing.assert_frame_equal(frames1[name], frames2[name])

    all_positive = _pairs(positives)
    observed = _pairs(frames1['observed_edges'])
    held_out = _pairs(frames1['test_random'].query('label == 1'))
    assert observed.isdisjoint(held_out)
    assert observed | held_out == all_positive
    assert len(observed) == round(len(all_positive) * 0.8)


def test_sparse_benchmark_negatives_are_verified_and_disjoint():
    positives, nodes = _source_graph()
    frames, _ = build_sparse_benchmark(positives, nodes, edge_retention=0.8, seed=42)
    positive = _pairs(positives)
    train_neg = _pairs(frames['train'].query('label == 0'))
    random_neg = _pairs(frames['test_random'].query('label == 0'))
    hard_neg = _pairs(frames['test_hard'].query('label == 0'))

    assert not (train_neg | random_neg | hard_neg) & positive
    assert train_neg.isdisjoint(random_neg)
    assert train_neg.isdisjoint(hard_neg)
    assert random_neg.isdisjoint(hard_neg)
    assert all(u < v for u, v in train_neg | random_neg | hard_neg)


def test_hard_negatives_share_an_observed_neighbour():
    positives, nodes = _source_graph()
    frames, _ = build_sparse_benchmark(positives, nodes, edge_retention=0.8, seed=42)
    graph = nx.from_pandas_edgelist(frames['observed_edges'], source='id1', target='id2')
    hard_neg = _pairs(frames['test_hard'].query('label == 0'))

    assert hard_neg
    assert all(set(graph.neighbors(u)) & set(graph.neighbors(v)) for u, v in hard_neg)


def test_sparse_pair_tables_are_balanced_and_indexed():
    positives, nodes = _source_graph()
    frames, meta = build_sparse_benchmark(positives, nodes, edge_retention=0.8, seed=42)

    for name in ('train', 'test_random', 'test_hard'):
        frame = frames[name]
        counts = frame.label.value_counts().to_dict()
        assert frame.index.name == 'id'
        assert counts[0] == counts[1]
    assert meta['protocol'] == 'wiki_sparse_holdout_v1'
    assert meta['counts']['held_out_positive_edges'] * 2 == len(frames['test_random'])


def test_mixed_training_changes_only_training_negatives():
    positives, nodes = _source_graph()
    random_frames, _ = build_sparse_benchmark(
        positives, nodes, edge_retention=0.8, seed=42, train_negatives='random')
    mixed_frames, metadata = build_sparse_benchmark(
        positives, nodes, edge_retention=0.8, seed=42, train_negatives='mixed')

    for name in ('observed_edges', 'test_random', 'test_hard'):
        pd.testing.assert_frame_equal(random_frames[name], mixed_frames[name])
    assert _pairs(random_frames['train'].query('label == 1')) == \
        _pairs(mixed_frames['train'].query('label == 1'))
    assert _pairs(random_frames['train'].query('label == 0')) != \
        _pairs(mixed_frames['train'].query('label == 0'))
    counts = metadata['counts']
    assert counts['train_random_negatives'] + counts['train_hard_negatives'] == \
        counts['observed_positive_edges']
    assert abs(counts['train_random_negatives'] - counts['train_hard_negatives']) == 1


def test_mixed_hard_training_negatives_are_verified_two_hop_pairs():
    positives, nodes = _source_graph()
    frames, metadata = build_sparse_benchmark(
        positives, nodes, edge_retention=0.8, seed=42, train_negatives='mixed')
    graph = nx.from_pandas_edgelist(frames['observed_edges'], source='id1', target='id2')
    full_positive = _pairs(positives)
    mixed_negatives = _pairs(frames['train'].query('label == 0'))
    two_hop = {
        pair for pair in mixed_negatives
        if set(graph.neighbors(pair[0])) & set(graph.neighbors(pair[1]))
    }

    assert not mixed_negatives & full_positive
    assert len(two_hop) >= metadata['counts']['train_hard_negatives']
