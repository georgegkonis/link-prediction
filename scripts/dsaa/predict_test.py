"""
Run a trained model on the test set and produce a Kaggle submission CSV.

Usage:
    python -m scripts.dsaa.predict_test model=structural
"""

import logging
import os
import numpy as np
import pandas as pd

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

log = logging.getLogger(__name__)

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

_MODEL_CLS = {
    'structural': StructuralClassifier,
    'tfidf':      TfidfClassifier,
    'pos':        PosClassifier,
    'embedding':  EmbeddingClassifier,
    'svm':        SvmClassifier,
    'cascade':    CascadeLP,
}


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    model_name = cfg.model.name
    raw_path = to_absolute_path(cfg.paths.raw)
    interim_path = to_absolute_path(cfg.paths.interim)
    predictions_path = to_absolute_path(cfg.paths.predictions)

    ckpt_path = cfg.get('eval_checkpoint')
    if not ckpt_path:
        ckpt_path = to_absolute_path(f'{cfg.paths.checkpoints}/{model_name}.joblib')
    else:
        ckpt_path = to_absolute_path(ckpt_path)

    os.makedirs(predictions_path, exist_ok=True)

    log.info('Loading model [%s] from %s...', model_name, ckpt_path)
    model = _MODEL_CLS[model_name].load(ckpt_path)

    test = load_edges(f'{raw_path}/test.csv')

    log.info('Loading test features...')
    if model_name in ('structural', 'cascade'):
        structural = pd.read_csv(f'{interim_path}/structural_test.csv', index_col='id')
    if model_name in ('tfidf', 'svm', 'cascade'):
        tfidf_scores = pd.read_csv(
            f'{interim_path}/tfidf_test.csv', index_col='id')['tfidf_score'].values
    if model_name in ('pos', 'cascade'):
        pos_features = np.load(f'{interim_path}/pos_test.npy')
    if model_name in ('embedding', 'cascade'):
        st_scores = pd.read_csv(
            f'{interim_path}/sentence_emb_test.csv', index_col='id')['st_score'].values

    log.info('Running inference...')
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
            y_pred, tier_used, y_scores = model.predict(structural, pos_features, st_scores, test)
            log.info('Tier usage on test set:')
            for tier, stats in model.tier_stats(tier_used).items():
                log.info('  %s: %s pairs (%.1f%%)', tier, f"{stats['n']:,}", stats['pct'])

    log.info('Inference time: %.1f ms  (%.3f ms/pair)', t[0], t[0] / len(test))

    tag = cfg.get('tag')
    name = f'{model_name}_{tag}' if tag else model_name

    if model_name == 'cascade':
        cn_thr, tfidf_thr = cfg.training.cn_threshold, cfg.training.tfidf_threshold
        if cn_thr is None or tfidf_thr is None:
            train = load_edges(f'{raw_path}/train.csv')
            cn_train = pd.read_csv(f'{interim_path}/structural_train.csv', index_col='id')['cn'].values
            tfidf_train = pd.read_csv(f'{interim_path}/tfidf_train.csv', index_col='id')['tfidf_score'].values
            auto_cn, auto_tfidf = pick_thresholds(train['label'].values, cn_train, tfidf_train)
            cn_thr    = cn_thr if cn_thr is not None else auto_cn
            tfidf_thr = tfidf_thr if tfidf_thr is not None else auto_tfidf

        difficulty = label_difficulty(
            test, structural['cn'].values, tfidf_scores, cn_thr, tfidf_thr)

        pd.DataFrame({
            'id': test.index, 'id1': test['id1'].values, 'id2': test['id2'].values,
            'y_pred': y_pred, 'tier_used': tier_used, 'difficulty': difficulty.values,
            'score': y_scores,
        }).to_csv(f'{predictions_path}/{name}_test_tiers.csv', index=False)
        log.info('Saved → %s/%s_test_tiers.csv', predictions_path, name)

    out = f'{predictions_path}/{name}_submission.csv'
    submission = pd.DataFrame({'id': test.index, 'label': y_pred})
    submission.to_csv(out, index=False)
    log.info('Saved → %s  (%s rows)', out, f'{len(submission):,}')


if __name__ == '__main__':
    main()
