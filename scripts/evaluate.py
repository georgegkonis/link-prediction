"""
Run a trained model on the test set and produce a Kaggle submission CSV.

Usage:
    python -m scripts.evaluate --model structural
    python -m scripts.evaluate --model cascade
"""

import argparse

import numpy as np
import pandas as pd

from src.data.loader import load_edges
from src.models.cascade import CascadeLP
from src.models.svm import (
    EmbeddingClassifier,
    PosClassifier,
    StructuralClassifier,
    TfidfClassifier,
)
from src.utils.metrics import timer

INTERIM     = 'data/interim'
CHECKPOINTS = 'outputs/checkpoints'
PREDICTIONS = 'outputs/predictions'

_MODEL_CLS = {
    'structural': StructuralClassifier,
    'tfidf':      TfidfClassifier,
    'pos':        PosClassifier,
    'embedding':  EmbeddingClassifier,
    'cascade':    CascadeLP,
}


def main(model_name: str):
    print(f'Loading model [{model_name}]...')
    model = _MODEL_CLS[model_name].load(f'{CHECKPOINTS}/{model_name}.joblib')

    test = load_edges('data/raw/test.csv')

    print('Loading test features...')
    if model_name in ('structural', 'cascade'):
        structural = pd.read_csv(f'{INTERIM}/structural_test.csv', index_col='id')
    if model_name == 'tfidf':
        tfidf_scores = pd.read_csv(
            f'{INTERIM}/tfidf_test.csv', index_col='id')['tfidf_score'].values
    if model_name in ('pos', 'cascade'):
        pos_features = np.load(f'{INTERIM}/pos_test.npy')
    if model_name in ('embedding', 'cascade'):
        st_scores = pd.read_csv(
            f'{INTERIM}/sentence_emb_test.csv', index_col='id')['st_score'].values

    print('Running inference...')
    with timer() as t:
        if model_name == 'structural':
            y_pred = model.predict(structural)
        elif model_name == 'tfidf':
            y_pred = model.predict(tfidf_scores)
        elif model_name == 'pos':
            y_pred = model.predict(pos_features)
        elif model_name == 'embedding':
            y_pred = model.predict(st_scores)
        elif model_name == 'cascade':
            y_pred, tier_used = model.predict(structural, pos_features, st_scores, test)
            print('\nTier usage on test set:')
            for tier, stats in model.tier_stats(tier_used).items():
                print(f"  {tier}: {stats['n']:,} pairs ({stats['pct']:.1f}%)")

    print(f'Inference time: {t[0]:.1f} ms  ({t[0] / len(test):.3f} ms/pair)')

    out = f'{PREDICTIONS}/{model_name}_submission.csv'
    submission = pd.DataFrame({'id': test.index, 'label': y_pred})
    submission.to_csv(out, index=False)
    print(f'Saved → {out}  ({len(submission):,} rows)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=list(_MODEL_CLS))
    args = p.parse_args()
    main(args.model)
