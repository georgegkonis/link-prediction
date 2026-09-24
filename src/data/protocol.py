"""Versioned pair holdout; no feature computation occurs in this module."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROTOCOL = 'graph_holdout_v1'
TEXT_SCOPE = 'known_corpus_transductive_unlabelled'


def fingerprint(pairs):
    return hashlib.sha256(pd.util.hash_pandas_object(
        pairs[[c for c in ['id1', 'id2', 'label'] if c in pairs]], index=True).values.tobytes()).hexdigest()


def pair_keys(pairs):
    return pd.MultiIndex.from_arrays([
        np.minimum(pairs.id1, pairs.id2), np.maximum(pairs.id1, pairs.id2)])


def grouped_split(pairs, val_size=0.2, seed=42):
    if not pairs.index.is_unique or not pairs.label.isin([0, 1]).all():
        raise ValueError('Expected unique pair IDs and binary labels')
    nonself = pairs.id1 != pairs.id2
    keys = pair_keys(pairs)
    groups = pd.DataFrame({'label': pairs.label.to_numpy()}, index=keys)
    if (groups.groupby(level=[0, 1]).label.nunique() > 1).any():
        raise ValueError('Conflicting labels within an unordered pair group')
    unique = groups.loc[nonself.to_numpy()].groupby(level=[0, 1]).label.first()
    _, val_groups = train_test_split(np.arange(len(unique)), test_size=val_size,
                                    stratify=unique.to_numpy(), random_state=seed)
    is_val = keys.isin(unique.index[val_groups])
    partition = np.where(nonself, np.where(is_val, 'val', 'train'), 'self_loop')
    return pd.Series(partition, index=pairs.index, name='partition')


def prepare_split(pairs, directory, val_size=0.2, seed=42):
    directory = Path(directory)
    if (directory / 'manifest.json').exists() or (directory / 'split.csv').exists():
        # Reusing a manifest is fine; silently replacing it is not.
        return load_split(pairs, directory, val_size, seed)
    partition = grouped_split(pairs, val_size, seed)
    directory.mkdir(parents=True, exist_ok=True)
    partition.to_csv(directory / 'split.csv')
    metadata = dict(protocol=PROTOCOL, pairs_sha256=fingerprint(pairs), seed=seed,
                    val_size=val_size, grouping='unordered_pair',
                    text_scope=TEXT_SCOPE, training_features='target_edge_masked',
                    graph='positive_train_partition_nonself_edges',
                    counts={k: int(v) for k, v in partition.value_counts().items()})
    metadata['split_sha256'] = hashlib.sha256((directory / 'split.csv').read_bytes()).hexdigest()
    (directory / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return partition, metadata


def load_split(pairs, directory, val_size=None, seed=None):
    directory = Path(directory)
    metadata = json.loads((directory / 'manifest.json').read_text())
    if metadata['protocol'] != PROTOCOL or metadata['pairs_sha256'] != fingerprint(pairs):
        raise ValueError('Split manifest does not match protocol or input pairs')
    for key, value in [('seed', seed), ('val_size', val_size)]:
        if value is not None and metadata[key] != value:
            raise ValueError(f'Split manifest {key} differs from requested configuration')
    if hashlib.sha256((directory / 'split.csv').read_bytes()).hexdigest() != metadata['split_sha256']:
        raise ValueError('Split file checksum mismatch')
    partition = pd.read_csv(directory / 'split.csv', index_col='id')['partition']
    if not partition.index.equals(pairs.index):
        raise ValueError('Split pair IDs/order differ from input')
    if not partition.isin(['train', 'val', 'self_loop']).all():
        raise ValueError('Unknown split partition')
    keys = pair_keys(pairs)
    if len(keys[partition == 'train'].intersection(keys[partition == 'val'])):
        raise ValueError('Unordered pair appears in both partitions')
    if not np.array_equal(partition == 'self_loop', pairs.id1 == pairs.id2):
        raise ValueError('Self-loop partition mismatch')
    return partition, metadata


def load_structural(pairs, directory, metadata, split='train'):
    directory = Path(directory)
    if not (directory / 'features.json').exists():
        raise FileNotFoundError(
            f'Revised structural features are missing from {directory}. Generate features for this '
            'split before loading them; legacy DSAA caches cannot be used.')
    feature_meta = json.loads((directory / 'features.json').read_text())
    if feature_meta['split'] != metadata:
        raise ValueError('Structural features belong to a different split/protocol')
    if split == 'test' and feature_meta['test_pairs_sha256'] != fingerprint(pairs):
        raise ValueError('Test pairs differ from structural feature provenance')
    path = directory / f'structural_{split}.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != feature_meta[f'{split}_features_sha256']:
        raise ValueError('Structural feature checksum mismatch')
    features = pd.read_csv(path, index_col='id')
    if not features.index.equals(pairs.index):
        raise ValueError('Structural feature IDs/order differ from input')
    return features
