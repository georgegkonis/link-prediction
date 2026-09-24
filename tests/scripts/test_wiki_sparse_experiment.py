import json

import pandas as pd
import pytest

from scripts.wiki.build_dataset import _write_sparse
from scripts.wiki.run_experiment import _load_sparse_benchmark


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
    _write_sparse(source, output, edge_retention=0.8, seed=42)

    manifest, frames, nodes = _load_sparse_benchmark(output / 'benchmark.json')

    assert manifest['protocol'] == 'wiki_sparse_holdout_v1'
    assert nodes == source / 'nodes.tsv'
    assert set(frames) == {'observed_edges', 'train', 'test_random', 'test_hard'}
    assert len(frames['test_random']) == len(frames['test_hard'])


def test_sparse_manifest_rejects_modified_pair_file(tmp_path):
    source = tmp_path / 'source'
    output = tmp_path / 'benchmark'
    _write_source(source)
    _write_sparse(source, output, edge_retention=0.8, seed=42)
    with (output / 'test_random.csv').open('a') as stream:
        stream.write('999,1,2,0\n')

    with pytest.raises(ValueError, match='file mismatch'):
        _load_sparse_benchmark(output / 'benchmark.json')


def test_sparse_manifest_records_file_hashes(tmp_path):
    source = tmp_path / 'source'
    output = tmp_path / 'benchmark'
    _write_source(source)
    _write_sparse(source, output, edge_retention=0.8, seed=42)

    manifest = json.loads((output / 'benchmark.json').read_text())
    assert set(manifest['files']) == {
        'observed_edges', 'train', 'test_random', 'test_hard'}
    assert all(len(item['sha256']) == 64 for item in manifest['files'].values())
