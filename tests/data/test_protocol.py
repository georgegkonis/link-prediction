import json
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from src.data.protocol import prepare_split, load_split, pair_keys
from src.features.structural import compute_heuristics


def pairs():
    frame = pd.DataFrame({'id1': np.arange(40), 'id2': np.arange(40) + 100,
                          'label': np.arange(40) % 2})
    reverse = frame.iloc[:10].rename(columns={'id1': 'id2', 'id2': 'id1'})
    frame = pd.concat([frame, reverse, pd.DataFrame({'id1': [999], 'id2': [999], 'label': [1]})], ignore_index=True)
    frame.index.name = 'id'
    return frame


def test_grouped_manifest_roundtrip_and_no_reciprocal_overlap(tmp_path):
    p = pairs()
    split, meta = prepare_split(p, tmp_path / 'split')
    assert split.iloc[-1] == 'self_loop'
    assert not len(pair_keys(p[split == 'train']).intersection(pair_keys(p[split == 'val'])))
    loaded, again = load_split(p, tmp_path / 'split')
    pd.testing.assert_series_equal(loaded, split)
    assert meta == again
    with pytest.raises(ValueError, match='seed'):
        prepare_split(p, tmp_path / 'split', seed=7)
    with pytest.raises(ValueError, match='input pairs'):
        load_split(p.iloc[::-1], tmp_path / 'split')


def test_manifest_rejects_tampering_and_label_conflicts(tmp_path):
    p = pairs()
    prepare_split(p, tmp_path / 'split')
    path = tmp_path / 'split' / 'split.csv'
    path.write_text(path.read_text().replace('train', 'val', 1))
    with pytest.raises(ValueError, match='checksum'):
        load_split(p, tmp_path / 'split')
    p.loc[40, 'label'] = 1
    with pytest.raises(ValueError, match='Conflicting'):
        prepare_split(p, tmp_path / 'other')


def test_training_target_cannot_supply_its_own_structural_evidence():
    graph = nx.Graph()
    graph.add_edge(1, 2, weight=7)
    graph.add_edge(3, 4)
    p = pd.DataFrame({'id1': [1, 2, 1, 1], 'id2': [2, 1, 3, 1]})
    before = graph.copy()
    actual = compute_heuristics(graph, p, show_progress=False, exclude_target_edge=True)
    assert (actual.iloc[:2] == 0).all().all()
    assert actual.iloc[2].pref_attach == 1
    assert actual.iloc[3].isna().all()
    assert nx.utils.graphs_equal(graph, before)
