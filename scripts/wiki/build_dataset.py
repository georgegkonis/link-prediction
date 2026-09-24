"""Build labelled pair datasets from the verified Wiki-CS-8k graph.

The default ``balanced`` protocol preserves the original experiment. The
``sparse-holdout`` protocol retains a fraction of real links as the observed
training graph and evaluates every remaining link against separate random and
two-hop hard-negative suites.
"""
import argparse
import json
import pathlib

import networkx as nx
import numpy as np
import pandas as pd

from src.data.feature_cache import sha256_file
from src.utils.log_utils import setup_logging

log = setup_logging('wiki_build_dataset')
SPARSE_PROTOCOL = 'wiki_sparse_holdout_v1'


def _canonical_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return unique non-self-loop pairs with ``id1 < id2``."""
    values = frame[['id1', 'id2']].to_numpy(dtype=np.int64)
    canonical = pd.DataFrame({
        'id1': np.minimum(values[:, 0], values[:, 1]),
        'id2': np.maximum(values[:, 0], values[:, 1]),
    })
    return canonical[canonical.id1 != canonical.id2].drop_duplicates().reset_index(drop=True)


def _pair_set(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return set(frame[['id1', 'id2']].itertuples(index=False, name=None))


def _sample_random_nonedges(
    node_ids: np.ndarray,
    positives: set[tuple[int, int]],
    n: int,
    rng: np.random.Generator,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    """Sample verified nonedges without replacement."""
    forbidden = set() if forbidden is None else set(forbidden)
    sampled: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(100_000, n * 100)
    while len(sampled) < n and attempts < max_attempts:
        batch = min(max(2 * (n - len(sampled)), 1_000), 1_000_000)
        left = rng.choice(node_ids, size=batch)
        right = rng.choice(node_ids, size=batch)
        for u, v in zip(left.tolist(), right.tolist()):
            if u == v:
                continue
            pair = (int(min(u, v)), int(max(u, v)))
            if pair in positives or pair in forbidden or pair in sampled:
                continue
            sampled.add(pair)
            if len(sampled) == n:
                break
        attempts += batch
    if len(sampled) != n:
        raise RuntimeError(f'Could only sample {len(sampled):,}/{n:,} random nonedges')
    return sampled


def _sample_hard_nonedges(
    observed: pd.DataFrame,
    positives: set[tuple[int, int]],
    n: int,
    rng: np.random.Generator,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    """Sample verified nonedges whose endpoints share an observed neighbour."""
    graph = nx.from_pandas_edgelist(observed, source='id1', target='id2')
    eligible = np.asarray([node for node, degree in graph.degree() if degree >= 2], dtype=np.int64)
    if not len(eligible):
        raise RuntimeError('Observed graph has no node with two neighbours; hard negatives are unavailable')
    neighbours = {node: np.asarray(list(graph.neighbors(node)), dtype=np.int64) for node in eligible}
    weights = np.asarray([
        len(neighbours[node]) * (len(neighbours[node]) - 1) / 2 for node in eligible
    ], dtype=float)
    weights /= weights.sum()
    forbidden = set() if forbidden is None else set(forbidden)
    sampled: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(250_000, n * 250)
    while len(sampled) < n and attempts < max_attempts:
        batch = min(max(2 * (n - len(sampled)), 1_000), 500_000)
        centres = rng.choice(eligible, size=batch, p=weights)
        for centre_value in centres:
            adjacent = neighbours[int(centre_value)]
            i = int(rng.integers(len(adjacent)))
            j = int(rng.integers(len(adjacent) - 1))
            if j >= i:
                j += 1
            u, v = int(adjacent[i]), int(adjacent[j])
            pair = (min(u, v), max(u, v))
            if pair not in positives and pair not in forbidden:
                sampled.add(pair)
                if len(sampled) == n:
                    break
        attempts += batch
    if len(sampled) != n:
        raise RuntimeError(f'Could only sample {len(sampled):,}/{n:,} hard nonedges')
    return sampled


def _labelled(positives: pd.DataFrame, negatives: set[tuple[int, int]], seed: int) -> pd.DataFrame:
    pos = positives[['id1', 'id2']].copy()
    pos['label'] = 1
    neg = pd.DataFrame(sorted(negatives), columns=['id1', 'id2'])
    neg['label'] = 0
    frame = pd.concat([pos, neg], ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    frame.index.name = 'id'
    return frame


def build_sparse_benchmark(
    positives: pd.DataFrame,
    node_ids: np.ndarray,
    edge_retention: float = 0.2,
    seed: int = 42,
    train_negatives: str = 'random',
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Create one training set and random/hard held-out evaluation suites."""
    if not 0 < edge_retention < 1:
        raise ValueError('edge_retention must be strictly between 0 and 1')
    if train_negatives not in {'random', 'mixed'}:
        raise ValueError("train_negatives must be 'random' or 'mixed'")
    positives = _canonical_pairs(positives)
    if positives.empty:
        raise ValueError('At least one positive edge is required')
    node_ids = np.asarray(sorted(set(map(int, node_ids))), dtype=np.int64)
    positive_set = _pair_set(positives)
    missing = set(positives.id1).union(positives.id2).difference(node_ids.tolist())
    if missing:
        raise ValueError(f'{len(missing)} positive-edge nodes are absent from nodes.tsv')

    rng = np.random.default_rng(seed)
    n_observed = int(round(len(positives) * edge_retention))
    observed_idx = np.sort(rng.choice(len(positives), size=n_observed, replace=False))
    observed_mask = np.zeros(len(positives), dtype=bool)
    observed_mask[observed_idx] = True
    observed = positives.loc[observed_mask].reset_index(drop=True)
    held_out = positives.loc[~observed_mask].reset_index(drop=True)

    base_train_neg = _sample_random_nonedges(node_ids, positive_set, len(observed), rng)
    random_test_neg = _sample_random_nonedges(
        node_ids, positive_set, len(held_out), rng, forbidden=base_train_neg)
    hard_test_neg = _sample_hard_nonedges(
        observed, positive_set, len(held_out), rng,
        forbidden=base_train_neg | random_test_neg)

    if train_negatives == 'mixed':
        n_random = (len(observed) + 1) // 2
        candidates = sorted(base_train_neg)
        chosen = np.sort(rng.choice(len(candidates), size=n_random, replace=False))
        train_random_neg = {candidates[index] for index in chosen}
        train_hard_neg = _sample_hard_nonedges(
            observed, positive_set, len(observed) - n_random, rng,
            forbidden=train_random_neg | random_test_neg | hard_test_neg)
        train_neg = train_random_neg | train_hard_neg
    else:
        train_random_neg = base_train_neg
        train_hard_neg = set()
        train_neg = base_train_neg

    graph = nx.from_pandas_edgelist(observed, source='id1', target='id2')
    frames = {
        'observed_edges': observed,
        'train': _labelled(observed, train_neg, seed + 1),
        'test_random': _labelled(held_out, random_test_neg, seed + 2),
        'test_hard': _labelled(held_out, hard_test_neg, seed + 3),
    }
    metadata = {
        'protocol': SPARSE_PROTOCOL,
        'seed': seed,
        'edge_retention': edge_retention,
        'undirected': True,
        'negative_sampling': {
            'train': ('half_uniform_half_observed_two_hop_verified_nonedge'
                      if train_negatives == 'mixed' else 'uniform_verified_nonedge'),
            'test_random': 'uniform_verified_nonedge',
            'test_hard': 'verified_nonedge_with_observed_common_neighbour',
        },
        'counts': {
            'nodes': int(len(node_ids)),
            'all_positive_edges': int(len(positives)),
            'observed_positive_edges': int(len(observed)),
            'held_out_positive_edges': int(len(held_out)),
            'train_random_negatives': int(len(train_random_neg)),
            'train_hard_negatives': int(len(train_hard_neg)),
            'train_pairs': int(len(frames['train'])),
            'test_random_pairs': int(len(frames['test_random'])),
            'test_hard_pairs': int(len(frames['test_hard'])),
        },
        'observed_graph': {
            'nodes': int(graph.number_of_nodes()),
            'edges': int(graph.number_of_edges()),
            'mean_degree': 2 * graph.number_of_edges() / max(graph.number_of_nodes(), 1),
        },
    }
    return frames, metadata


def _write_sparse(source: pathlib.Path, output: pathlib.Path, edge_retention: float,
                  seed: int, train_negatives: str = 'random') -> None:
    positives_path = source / 'positive_edges.csv'
    nodes_path = source / 'nodes.tsv'
    positives = pd.read_csv(positives_path)
    node_ids = pd.read_csv(nodes_path, sep='\t', usecols=['id'])['id'].to_numpy()
    frames, metadata = build_sparse_benchmark(
        positives, node_ids, edge_retention, seed, train_negatives=train_negatives)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / f'{name}.csv' for name in frames}
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=name != 'observed_edges')
    metadata['source'] = {
        'positive_edges': str(positives_path),
        'positive_edges_sha256': sha256_file(positives_path),
        'nodes': str(nodes_path),
        'nodes_sha256': sha256_file(nodes_path),
    }
    metadata['files'] = {
        name: {'path': str(path), 'sha256': sha256_file(path)} for name, path in paths.items()
    }
    manifest = output / 'benchmark.json'
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    log.info('Wrote sparse benchmark manifest -> %s', manifest)
    log.info('Observed graph: %s edges, mean degree %.2f; held-out positives: %s',
             f"{metadata['observed_graph']['edges']:,}", metadata['observed_graph']['mean_degree'],
             f"{metadata['counts']['held_out_positive_edges']:,}")


def _write_balanced(output: pathlib.Path, neg_ratio: float, seed: int) -> None:
    positives = _canonical_pairs(pd.read_csv(output / 'positive_edges.csv'))
    nodes = pd.read_csv(output / 'nodes.tsv', sep='\t', index_col='id')
    positive_pairs = _pair_set(positives)
    n_neg = int(round(len(positives) * neg_ratio))
    negatives = _sample_random_nonedges(
        nodes.index.to_numpy(), positive_pairs, n_neg, np.random.default_rng(seed))
    dataset = _labelled(positives, negatives, seed)
    dataset.to_csv(output / 'train.csv')
    log.info('Wrote %s rows (%s positive, %s negative) -> %s',
             f'{len(dataset):,}', f'{len(positives):,}', f'{len(negatives):,}', output / 'train.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=('balanced', 'sparse-holdout'), default='balanced')
    parser.add_argument('--source', default='data/raw/wiki_cs_8k')
    parser.add_argument('--output')
    parser.add_argument('--neg-ratio', type=float, default=1.0)
    parser.add_argument('--edge-retention', type=float, default=0.2)
    parser.add_argument('--train-negatives', choices=('random', 'mixed'), default='random')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source = pathlib.Path(args.source)
    sparse_output = ('data/raw/wiki_cs_8k_sparse20_mixed'
                     if args.train_negatives == 'mixed' else 'data/raw/wiki_cs_8k_sparse20')
    output = pathlib.Path(args.output or (
        sparse_output if args.protocol == 'sparse-holdout' else args.source))
    if args.protocol == 'sparse-holdout':
        _write_sparse(source, output, args.edge_retention, args.seed, args.train_negatives)
    else:
        _write_balanced(output, args.neg_ratio, args.seed)


if __name__ == '__main__':
    main()
