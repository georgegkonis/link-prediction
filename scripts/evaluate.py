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
    SvmClassifier,
    TfidfClassifier,
)
from src.utils.difficulty import label_difficulty, pick_thresholds
from src.utils.metrics import timer

INTERIM     = 'data/interim'
CHECKPOINTS = 'outputs/checkpoints'
PREDICTIONS = 'outputs/predictions'

_MODEL_CLS = {
    'structural': StructuralClassifier,
    'tfidf':      TfidfClassifier,
    'pos':        PosClassifier,
    'embedding':  EmbeddingClassifier,
    'svm':        SvmClassifier,
    'cascade':    CascadeLP,
}


def main(model_name: str, cn_threshold: float | None, tfidf_threshold: float | None):
    print(f'Loading model [{model_name}]...')
    model = _MODEL_CLS[model_name].load(f'{CHECKPOINTS}/{model_name}.joblib')

    test = load_edges('data/raw/test.csv')

    print('Loading test features...')
    if model_name in ('structural', 'cascade'):
        structural = pd.read_csv(f'{INTERIM}/structural_test.csv', index_col='id')
    if model_name in ('tfidf', 'svm', 'cascade'):
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
        elif model_name == 'svm':
            y_pred = model.predict(tfidf_scores)
        elif model_name == 'cascade':
            y_pred, tier_used = model.predict(structural, pos_features, st_scores, test)
            print('\nTier usage on test set:')
            for tier, stats in model.tier_stats(tier_used).items():
                print(f"  {tier}: {stats['n']:,} pairs ({stats['pct']:.1f}%)")

    print(f'Inference time: {t[0]:.1f} ms  ({t[0] / len(test):.3f} ms/pair)')

    if model_name == 'cascade':
        cn_thr, tfidf_thr = cn_threshold, tfidf_threshold
        if cn_thr is None or tfidf_thr is None:
            train = load_edges('data/raw/train.csv')
            cn_train = pd.read_csv(f'{INTERIM}/structural_train.csv', index_col='id')['cn'].values
            tfidf_train = pd.read_csv(f'{INTERIM}/tfidf_train.csv', index_col='id')['tfidf_score'].values
            auto_cn, auto_tfidf = pick_thresholds(train['label'].values, cn_train, tfidf_train)
            cn_thr    = cn_thr if cn_thr is not None else auto_cn
            tfidf_thr = tfidf_thr if tfidf_thr is not None else auto_tfidf

        difficulty = label_difficulty(
            test, structural['cn'].values, tfidf_scores, cn_thr, tfidf_thr)

        pd.DataFrame({
            'id': test.index, 'id1': test['id1'].values, 'id2': test['id2'].values,
            'y_pred': y_pred, 'tier_used': tier_used, 'difficulty': difficulty.values,
        }).to_csv(f'{PREDICTIONS}/cascade_test_tiers.csv', index=False)
        print(f'Saved → {PREDICTIONS}/cascade_test_tiers.csv')

    out = f'{PREDICTIONS}/{model_name}_submission.csv'
    submission = pd.DataFrame({'id': test.index, 'label': y_pred})
    submission.to_csv(out, index=False)
    print(f'Saved → {out}  ({len(submission):,} rows)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=list(_MODEL_CLS))
    p.add_argument('--cn-threshold',    type=float, default=None)
    p.add_argument('--tfidf-threshold', type=float, default=None)
    args = p.parse_args()
    main(args.model, args.cn_threshold, args.tfidf_threshold)
