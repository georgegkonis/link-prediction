"""
Ablate node2vec embeddings to evaluate their impact.

Usage:
    python -m scripts.analysis.ablate_node2vec
"""

import json
import pathlib

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_edges
from src.features.structural import node2vec_hadamard_features
from src.utils.log_utils import setup_logging

INTERIM = 'data/interim/dsaa'
RAW = 'data/raw/dsaa'
PREDICTIONS = 'outputs/predictions/dsaa'
HEURISTICS = ['cn', 'jaccard', 'adamic_adar', 'pref_attach']
TIER1_THRESHOLD = 0.8


def fit_and_score(X_tr, y_tr, X_val, y_val):
    scaler = StandardScaler()
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(scaler.fit_transform(X_tr), y_tr)
    proba = clf.predict_proba(scaler.transform(X_val))
    y_pred = proba.argmax(axis=1)
    confident = proba.max(axis=1) >= TIER1_THRESHOLD
    return {
        'n': len(y_val),
        'accuracy': accuracy_score(y_val, y_pred),
        'macro_f1': f1_score(y_val, y_pred, average='macro', zero_division=0),
        'tier1_call_rate_%': 100 * confident.mean(),
        'tier1_call_n': int(confident.sum()),
        'accuracy_confident': accuracy_score(y_val[confident], y_pred[confident]) if confident.any() else float('nan'),
        'macro_f1_confident': f1_score(y_val[confident], y_pred[confident], average='macro', zero_division=0) if confident.any() else float('nan'),
    }


def coverage_stats(wv, test: pd.DataFrame, hadamard_test: np.ndarray) -> dict:
    """Fraction of test.csv id1/id2 present in the (transductive) Node2Vec vocabulary."""
    vocab = wv.key_to_index
    id1_covered = test['id1'].astype(str).isin(vocab)
    id2_covered = test['id2'].astype(str).isin(vocab)
    zero_vec = ~(hadamard_test != 0).any(axis=1)
    return {
        'id1_coverage_pct': 100 * id1_covered.mean(),
        'id2_coverage_pct': 100 * id2_covered.mean(),
        'zero_vec_pct': 100 * zero_vec.mean(),
        'zero_vec_n': int(zero_vec.sum()),
        'test_n': len(test),
    }


def main():
    log = setup_logging('ablate_node2vec')
    train = load_edges(f'{RAW}/train.csv')
    y = train['label'].values
    structural = pd.read_csv(f'{INTERIM}/structural_train.csv', index_col='id')

    not_self = (train['id1'] != train['id2']).values
    idx = np.where(not_self)[0]
    tr, val = train_test_split(idx, test_size=0.2, stratify=y[idx], random_state=42)

    heuristics = structural[HEURISTICS].fillna(0).values

    log.info('Loading Node2Vec embeddings...')
    wv = KeyedVectors.load(f'{INTERIM}/node2vec.kv')
    hadamard = node2vec_hadamard_features(wv, train)

    without = fit_and_score(heuristics[tr], y[tr], heuristics[val], y[val])
    lines = ['--- Without Node2Vec (4-dim heuristics only) ---']
    for k, v in without.items():
        lines.append(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    log.info('\n'.join(lines))

    augmented_tr = np.hstack([heuristics[tr], hadamard[tr]])
    augmented_val = np.hstack([heuristics[val], hadamard[val]])
    with_n2v = fit_and_score(augmented_tr, y[tr], augmented_val, y[val])
    lines = ['--- With Node2Vec (4 heuristics + 64-dim Hadamard = 68-dim) ---']
    for k, v in with_n2v.items():
        lines.append(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    log.info('\n'.join(lines))

    delta = {k: with_n2v[k] - without[k] for k in without if isinstance(without[k], float)}
    lines = ['--- Delta (with - without) ---']
    for k, v in delta.items():
        lines.append(f'  {k}: {v:+.4f}')
    log.info('\n'.join(lines))

    log.info('Computing test-set Node2Vec vocabulary coverage...')
    test = load_edges(f'{RAW}/test.csv')
    hadamard_test = node2vec_hadamard_features(wv, test)
    coverage = coverage_stats(wv, test, hadamard_test)
    log.info('Coverage: %s', coverage)

    out = {'without': without, 'with': with_n2v, 'delta': delta, 'coverage': coverage}
    out_path = pathlib.Path(PREDICTIONS) / 'node2vec_ablation.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info('Saved → %s', out_path)


if __name__ == '__main__':
    main()
