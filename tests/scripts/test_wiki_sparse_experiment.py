import json

import joblib
import pandas as pd
import pytest

from scripts.wiki.run_experiment import _load_shared_node_cache, _load_sparse_benchmark
from src.data.feature_cache import save_cache_manifest
from wikilinkgen.pairs import write_sparse


def _write_source(path):
    path.mkdir()
    edges = [(i, i + 1) for i in range(20)]
    edges += [(i, i + 2) for i in range(0, 19, 2)]
    pd.DataFrame(edges, columns=['id1', 'id2']).to_csv(
        path / 'positive_edges.csv', index=False)
    pd.DataFrame({'id': range(21), 'text': [f'article {i}' for i in range(21)]}).to_csv(
        path / 'nodes.tsv', sep='\t', index=False)


def test_sparse_manifest_round_trip_validates_sources_and_partitions(tmp_path):
    source = tmp_path / 'source'
    output = tmp_path / 'benchmark'
    _write_source(source)
    write_sparse(source, output, edge_retention=0.8, seed=42)

    manifest, frames, nodes = _load_sparse_benchmark(output / 'benchmark.json')

    assert manifest['protocol'] == 'wiki_sparse_holdout_v1'
    assert nodes == source / 'nodes.tsv'
    assert set(frames) == {'observed_edges', 'train', 'test_random', 'test_hard'}
    assert len(frames['test_random']) == len(frames['test_hard'])


def test_sparse_manifest_rejects_modified_pair_file(tmp_path):
    source = tmp_path / 'source'
    output = tmp_path / 'benchmark'
    _write_source(source)
    write_sparse(source, output, edge_retention=0.8, seed=42)
    with (output / 'test_random.csv').open('a') as stream:
        stream.write('999,1,2,0\n')

    with pytest.raises(ValueError, match='file mismatch'):
        _load_sparse_benchmark(output / 'benchmark.json')


def test_sparse_manifest_records_file_hashes(tmp_path):
    source = tmp_path / 'source'
    output = tmp_path / 'benchmark'
    _write_source(source)
    write_sparse(source, output, edge_retention=0.8, seed=42)

    manifest = json.loads((output / 'benchmark.json').read_text())
    assert set(manifest['files']) == {
        'observed_edges', 'train', 'test_random', 'test_hard'}
    assert all(len(item['sha256']) == 64 for item in manifest['files'].values())


def _write_node_cache(path, request):
    path.mkdir()
    values = {
        'tfidf_vectorizer': 'vectorizer',
        'tfidf_nodes': ({1: 0, 2: 1, 3: 2}, 'matrix'),
        'pos_nodes': {1: 'pos-1', 2: 'pos-2', 3: 'pos-3'},
        'embedding_nodes': {1: 'emb-1', 2: 'emb-2', 3: 'emb-3'},
    }
    files = {name: path / f'{name}.joblib' for name in values}
    for name, value in values.items():
        joblib.dump(value, files[name])
    save_cache_manifest(path, request, files)
    return values, files


def test_shared_node_cache_requires_matching_provenance_and_nodes(tmp_path):
    request = {
        'nodes_sha256': 'nodes',
        'features': {'tfidf': {'min_df': 2}, 'embedding': {'model_name': 'model'}},
        'feature_code_sha256': {
            'structural': 'new-structural',
            'embeddings': 'embeddings',
            'linguistic': 'linguistic',
        },
    }
    source_request = {
        **request,
        'benchmark_manifest_sha256': 'source-benchmark',
        'feature_code_sha256': {**request['feature_code_sha256'], 'structural': 'old-structural'},
    }
    values, _ = _write_node_cache(tmp_path / 'cache', source_request)

    loaded = _load_shared_node_cache(tmp_path / 'cache', request, [1, 2])

    assert loaded == (
        values['tfidf_vectorizer'], values['tfidf_nodes'],
        values['pos_nodes'], values['embedding_nodes'])


def test_shared_node_cache_rejects_modified_file(tmp_path):
    request = {
        'nodes_sha256': 'nodes',
        'features': {},
        'feature_code_sha256': {'embeddings': 'e', 'linguistic': 'l'},
    }
    _, files = _write_node_cache(tmp_path / 'cache', request)
    files['pos_nodes'].write_bytes(b'changed')

    with pytest.raises(ValueError, match='file mismatch'):
        _load_shared_node_cache(tmp_path / 'cache', request, [1, 2])
