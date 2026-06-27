"""
Node2Vec ablation for CascadeLP Tier 1 (thesis Ch.5 SS5.1): compare the
StructuralClassifier's held-out validation performance with vs. without the
64-dim Node2Vec Hadamard-product features appended to the 4 structural
heuristics (per chap3.tex SS3.2 "Node2Vec Embeddings"), holding the split,
threshold, and heuristic features identical.

Usage:
    python -m scripts.ablate_node2vec
"""

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_edges
from src.features.structural import node2vec_hadamard_features

INTERIM = 'data/interim'
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
        'accuracy_confident': accuracy_score(y_val[confident], y_pred[confident]) if confident.any() else float('nan'),
        'macro_f1_confident': f1_score(y_val[confident], y_pred[confident], average='macro', zero_division=0) if confident.any() else float('nan'),
    }


def main():
    train = load_edges('data/raw/train.csv')
    y = train['label'].values
    structural = pd.read_csv(f'{INTERIM}/structural_train.csv', index_col='id')

    not_self = (train['id1'] != train['id2']).values
    idx = np.where(not_self)[0]
    tr, val = train_test_split(idx, test_size=0.2, stratify=y[idx], random_state=42)

    heuristics = structural[HEURISTICS].fillna(0).values

    print('Loading Node2Vec embeddings...')
    wv = KeyedVectors.load(f'{INTERIM}/node2vec.kv')
    hadamard = node2vec_hadamard_features(wv, train)

    print('\n--- Without Node2Vec (4-dim heuristics only) ---')
    baseline = fit_and_score(heuristics[tr], y[tr], heuristics[val], y[val])
    for k, v in baseline.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    print('\n--- With Node2Vec (4 heuristics + 64-dim Hadamard = 68-dim) ---')
    augmented_tr = np.hstack([heuristics[tr], hadamard[tr]])
    augmented_val = np.hstack([heuristics[val], hadamard[val]])
    with_n2v = fit_and_score(augmented_tr, y[tr], augmented_val, y[val])
    for k, v in with_n2v.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    print('\n--- Delta (with - without) ---')
    for k in baseline:
        if isinstance(baseline[k], float):
            print(f'  {k}: {with_n2v[k] - baseline[k]:+.4f}')


if __name__ == '__main__':
    main()
