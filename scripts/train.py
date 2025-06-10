"""
Train and validate a model on precomputed features.

Available models:
    structural  — LogReg on CN/Jaccard/Adamic-Adar/PA heuristics
    tfidf       — LogReg on TF-IDF cosine similarity
    pos         — Random Forest on POS frequency features
    embedding   — LogReg on sentence-transformer cosine similarity
    cascade     — CascadeLP (all three tiers combined)

Usage:
    python -m scripts.train --model structural
    python -m scripts.train --model cascade --tier1-threshold 0.8 --tier2-threshold 0.7
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import build_graph, load_edges
from src.models.cascade import CascadeLP
from src.models.svm import (
    EmbeddingClassifier,
    PosClassifier,
    StructuralClassifier,
    TfidfClassifier,
)
from src.utils.metrics import cold_start_mask, evaluate, timer

INTERIM     = 'data/interim'
CHECKPOINTS = 'outputs/checkpoints'


def _load(model_name: str) -> dict:
    train = load_edges('data/raw/train.csv')
    data  = {'pairs': train, 'y': train['label'].values}

    if model_name in ('structural', 'cascade'):
        data['structural'] = pd.read_csv(f'{INTERIM}/structural_train.csv', index_col='id')

    if model_name == 'tfidf':
        data['tfidf_scores'] = pd.read_csv(
            f'{INTERIM}/tfidf_train.csv', index_col='id')['tfidf_score'].values

    if model_name in ('pos', 'cascade'):
        data['pos_features'] = np.load(f'{INTERIM}/pos_train.npy')

    if model_name in ('embedding', 'cascade'):
        data['st_scores'] = pd.read_csv(
            f'{INTERIM}/sentence_emb_train.csv', index_col='id')['st_score'].values

    return data


def _split(data: dict, model_name: str, val_size: float):
    pairs = data['pairs']
    y     = data['y']

    not_self = (pairs['id1'] != pairs['id2']).values
    idx      = np.where(not_self)[0]
    tr, val  = train_test_split(idx, test_size=val_size, stratify=y[idx], random_state=42)

    def sub(v):
        if isinstance(v, pd.DataFrame):
            return v.iloc[tr], v.iloc[val]
        return v[tr], v[val]

    return tr, val, sub


def main(model_name: str, t1: float, t2: float, val_size: float):
    print(f'Loading features for [{model_name}]...')
    data = _load(model_name)
    tr, val, sub = _split(data, model_name, val_size)
    y = data['y']

    print(f'Train: {len(tr):,}  |  Val: {len(val):,}')

    if model_name == 'structural':
        model = StructuralClassifier()
        tr_X, val_X = sub(data['structural'])
        model.fit(tr_X, y[tr])
        with timer() as t:
            proba = model.predict_proba(val_X)
        y_pred, y_scores = proba.argmax(axis=1), proba[:, 1]

    elif model_name == 'tfidf':
        model = TfidfClassifier()
        tr_X, val_X = sub(data['tfidf_scores'])
        model.fit(tr_X, y[tr])
        with timer() as t:
            proba = model.predict_proba(val_X)
        y_pred, y_scores = proba.argmax(axis=1), proba[:, 1]

    elif model_name == 'pos':
        model = PosClassifier()
        tr_X, val_X = sub(data['pos_features'])
        model.fit(tr_X, y[tr])
        with timer() as t:
            proba = model.predict_proba(val_X)
        y_pred, y_scores = proba.argmax(axis=1), proba[:, 1]

    elif model_name == 'embedding':
        model = EmbeddingClassifier()
        tr_X, val_X = sub(data['st_scores'])
        model.fit(tr_X, y[tr])
        with timer() as t:
            proba = model.predict_proba(val_X)
        y_pred, y_scores = proba.argmax(axis=1), proba[:, 1]

    elif model_name == 'cascade':
        model = CascadeLP(tier1_threshold=t1, tier2_threshold=t2)
        model.fit(
            data['structural'], data['pos_features'],
            data['st_scores'],  y, data['pairs'],
        )
        val_pairs = data['pairs'].iloc[val]
        with timer() as t:
            y_pred, tier_used = model.predict(
                data['structural'].iloc[val],
                data['pos_features'][val],
                data['st_scores'][val],
                val_pairs,
            )
        y_scores = y_pred.astype(float)
        print('\nTier usage:')
        for tier, stats in model.tier_stats(tier_used).items():
            print(f"  {tier}: {stats['n']:,} pairs ({stats['pct']:.1f}%)")

    G      = build_graph(data['pairs'])
    cs     = cold_start_mask(data['pairs'].iloc[val], G)
    result = evaluate(y[val], y_pred, y_scores, cs, latency_ms=t[0] if t else None)
    print(f'\nValidation — {model_name}')
    print(result)

    out = f'{CHECKPOINTS}/{model_name}.joblib'
    model.save(out)
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',            required=True,
                   choices=['structural', 'tfidf', 'pos', 'embedding', 'cascade'])
    p.add_argument('--tier1-threshold',  type=float, default=0.8)
    p.add_argument('--tier2-threshold',  type=float, default=0.7)
    p.add_argument('--val-size',         type=float, default=0.2)
    args = p.parse_args()
    main(args.model, args.tier1_threshold, args.tier2_threshold, args.val_size)
