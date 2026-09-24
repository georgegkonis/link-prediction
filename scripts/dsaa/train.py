"""
Train and validate a model on precomputed features using the Hydra configuration.

Reads:
    <paths.raw>/train.csv and model-specific files in <paths.interim>/
Writes:
    <paths.checkpoints>/<model[_tag]>.joblib
    <paths.predictions>/<model[_tag]>_val_metrics.json
    <paths.predictions>/<model[_tag]>_run_config.json

Usage:
    python -m scripts.dsaa.train model=cascade [tag=NAME]
"""

import json
import logging
import os
import pathlib

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import build_graph, load_edges
from src.data.run_config import save_run_config
from src.models.cascade import CascadeLP
from src.models.svm import (
    EmbeddingClassifier,
    PosClassifier,
    StructuralClassifier,
    SvmClassifier,
    TfidfClassifier,
)
from src.utils.difficulty import label_difficulty, pick_thresholds
from src.utils.metrics import (
    evaluate,
    evaluate_by_group,
    tier_difficulty_breakdown,
    timer,
    zero_common_neighbors_mask,
)

def _load(model_name: str, raw_path: str, interim_path: str) -> dict:
    train = load_edges(os.path.join(raw_path, 'train.csv'))
    data  = {'pairs': train, 'y': train['label'].values}

    def aligned_csv(filename: str) -> pd.DataFrame:
        frame = pd.read_csv(f'{interim_path}/{filename}', index_col='id')
        if not frame.index.equals(train.index):
            raise ValueError(f'Feature pair IDs do not match {raw_path}/train.csv: {filename}')
        return frame

    if model_name in ('structural', 'cascade'):
        data['structural'] = aligned_csv('structural_train.csv')

    if model_name == 'svm':
        # cn only, for per-pair difficulty labeling of the error-by-difficulty export below
        data['structural'] = aligned_csv('structural_train.csv')

    if model_name in ('tfidf', 'svm', 'cascade'):
        data['tfidf_scores'] = aligned_csv('tfidf_train.csv')['tfidf_score'].values

    if model_name in ('pos', 'cascade'):
        data['pos_features'] = np.load(f'{interim_path}/pos_train.npy')
        if len(data['pos_features']) != len(train):
            raise ValueError('POS feature row count does not match DSAA training pairs')

    if model_name in ('embedding', 'cascade'):
        data['st_scores'] = aligned_csv('sentence_emb_train.csv')['st_score'].values

    return data


def _split(data: dict, model_name: str, val_size: float, seed: int):
    pairs = data['pairs']
    y     = data['y']

    not_self = (pairs['id1'] != pairs['id2']).values
    idx      = np.where(not_self)[0]
    tr, val  = train_test_split(idx, test_size=val_size, stratify=y[idx], random_state=seed)

    def sub(v):
        if isinstance(v, pd.DataFrame):
            return v.iloc[tr], v.iloc[val]
        return v[tr], v[val]

    return tr, val, sub


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    model_name = cfg.model.name
    tag = cfg.get('tag')
    name = f'{model_name}_{tag}' if tag else model_name
    raw_path = to_absolute_path(cfg.paths.raw)
    interim_path = to_absolute_path(cfg.paths.interim)
    checkpoints_path = to_absolute_path(cfg.paths.checkpoints)
    predictions_path = to_absolute_path(cfg.paths.predictions)

    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(predictions_path, exist_ok=True)

    log.info(f'Loading features for [{model_name}]...')
    data = _load(model_name, raw_path, interim_path)
    tr, val, sub = _split(data, model_name, cfg.training.val_size, cfg.seed)
    y = data['y']

    log.info(f'Train: {len(tr):,}  |  Val: {len(val):,}')

    if model_name == 'structural':
        model = StructuralClassifier(
            C=cfg.model.get('C', 1.0), max_iter=cfg.model.get('max_iter', 1000), random_state=cfg.seed)
        tr_X, val_X = sub(data['structural'])
        model.fit(tr_X, y[tr])
        with timer() as t:
            proba = model.predict_proba(val_X)
        y_pred, y_scores = proba.argmax(axis=1), proba[:, 1]

    elif model_name == 'tfidf':
        model = TfidfClassifier(
            C=cfg.model.get('C', 1.0), max_iter=cfg.model.get('max_iter', 1000), random_state=cfg.seed)
        tr_X, val_X = sub(data['tfidf_scores'])
        model.fit(tr_X, y[tr])
        with timer() as t:
            proba = model.predict_proba(val_X)
        y_pred, y_scores = proba.argmax(axis=1), proba[:, 1]

    elif model_name == 'pos':
        model = PosClassifier(
            n_estimators=cfg.model.get('n_estimators', 200),
            class_weight=cfg.model.get('class_weight', 'balanced'),
            random_state=cfg.seed)
        tr_X, val_X = sub(data['pos_features'])
        model.fit(tr_X, y[tr])
        with timer() as t:
            proba = model.predict_proba(val_X)
        y_pred, y_scores = proba.argmax(axis=1), proba[:, 1]

    elif model_name == 'embedding':
        model = EmbeddingClassifier(
            C=cfg.model.get('C', 1.0), max_iter=cfg.model.get('max_iter', 1000), random_state=cfg.seed)
        tr_X, val_X = sub(data['st_scores'])
        model.fit(tr_X, y[tr])
        with timer() as t:
            proba = model.predict_proba(val_X)
        y_pred, y_scores = proba.argmax(axis=1), proba[:, 1]

    elif model_name == 'svm':
        model = SvmClassifier(
            C=cfg.model.get('C', 1.0), 
            gamma=cfg.model.get('gamma', 'scale'),
            subsample_size=cfg.model.get('subsample_size', 20000),
            random_state=cfg.seed
        )
        tr_X, val_X = sub(data['tfidf_scores'])
        model.fit(tr_X, y[tr])
        with timer() as t:
            proba = model.predict_proba(val_X)
        y_pred, y_scores = proba.argmax(axis=1), proba[:, 1]

        val_pairs = data['pairs'].iloc[val]
        cn_val, tfidf_val = data['structural']['cn'].values, data['tfidf_scores']
        cn_thr, tfidf_thr = cfg.training.cn_threshold, cfg.training.tfidf_threshold
        if cn_thr is None or tfidf_thr is None:
            auto_cn, auto_tfidf = pick_thresholds(y, cn_val, tfidf_val)
            cn_thr    = cn_thr if cn_thr is not None else auto_cn
            tfidf_thr = tfidf_thr if tfidf_thr is not None else auto_tfidf
        difficulty = label_difficulty(val_pairs, cn_val[val], tfidf_val[val], cn_thr, tfidf_thr)

        pd.DataFrame({
            'id': val_pairs.index, 'y_true': y[val], 'y_pred': y_pred,
            'correct': y_pred == y[val], 'difficulty': difficulty.values, 'score': y_scores,
        }).to_csv(f'{predictions_path}/{name}_val_errors.csv', index=False)
        log.info('Saved → %s/%s_val_errors.csv', predictions_path, name)

    elif model_name == 'cascade':
        model = CascadeLP(tier1_threshold=cfg.model.tier1_threshold, tier2_threshold=cfg.model.tier2_threshold)
        tr_structural, val_structural = sub(data['structural'])
        tr_pos, val_pos = sub(data['pos_features'])
        tr_st, val_st = sub(data['st_scores'])
        model.fit(tr_structural, tr_pos, tr_st, y[tr], data['pairs'].iloc[tr])
        val_pairs = data['pairs'].iloc[val]
        with timer() as t:
            y_pred, tier_used, y_scores = model.predict(
                val_structural, val_pos, val_st, val_pairs,
            )
        log.info('Tier usage:')
        for tier, stats in model.tier_stats(tier_used).items():
            log.info(f"  {tier}: {stats['n']:,} pairs ({stats['pct']:.1f}%)")

        cn_val, tfidf_val = data['structural']['cn'].values, data['tfidf_scores']
        cn_thr, tfidf_thr = cfg.training.cn_threshold, cfg.training.tfidf_threshold
        if cn_thr is None or tfidf_thr is None:
            auto_cn, auto_tfidf = pick_thresholds(y, cn_val, tfidf_val)
            cn_thr     = cn_thr if cn_thr is not None else auto_cn
            tfidf_thr  = tfidf_thr if tfidf_thr is not None else auto_tfidf

        difficulty = label_difficulty(
            val_pairs, cn_val[val], tfidf_val[val], cn_thr, tfidf_thr)

        log.info('Accuracy by tier:\n%s', evaluate_by_group(y[val], y_pred, tier_used))
        log.info('Accuracy by difficulty:\n%s', evaluate_by_group(y[val], y_pred, difficulty.values))
        log.info('Tier × difficulty breakdown:\n%s', tier_difficulty_breakdown(y[val], y_pred, tier_used, difficulty))

        pd.DataFrame({
            'id': val_pairs.index, 'id1': val_pairs['id1'].values, 'id2': val_pairs['id2'].values,
            'y_true': y[val], 'y_pred': y_pred, 'correct': y_pred == y[val],
            'tier_used': tier_used, 'difficulty': difficulty.values, 'score': y_scores,
        }).to_csv(f'{predictions_path}/{name}_val_tiers.csv', index=False)
        log.info('Saved → %s/%s_val_tiers.csv', predictions_path, name)

    G      = build_graph(data['pairs'])
    # Historical DSAA reports used the broad zero-CN diagnostic. Keep those
    # values reproducible while naming the population explicitly in code.
    zero_cn = zero_common_neighbors_mask(data['pairs'].iloc[val], G)
    result = evaluate(y[val], y_pred, y_scores, zero_cn, latency_ms=t[0] if t else None)
    log.info('Validation — %s\n%s', model_name, result)

    pathlib.Path(f'{predictions_path}/{name}_val_metrics.json').write_text(
        json.dumps({
            'macro_f1':      result.macro_f1,
            'auc_roc':       result.auc_roc,
            'cold_start_f1': result.cold_start_macro_f1,
            'latency_ms':    result.latency_ms,
            'n_val':         int(len(y[val])),
        }, indent=2)
    )

    out = f'{checkpoints_path}/{name}.joblib'
    model.save(out)
    log.info('Saved → %s', out)
    save_run_config(cfg, pathlib.Path(predictions_path) / f'{name}_run_config.json',
                    checkpoint=out, raw_path=raw_path, interim_path=interim_path,
                    n_train=len(tr), n_val=len(val))


if __name__ == '__main__':
    main()
