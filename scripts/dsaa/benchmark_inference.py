"""
Benchmark CPU inference throughput.

Reads:
    DSAA training pairs, cached features, and cascade.joblib
Writes:
    outputs/predictions/dsaa/throughput_benchmark.json

Usage:
    python -m scripts.dsaa.benchmark_inference [--predictions PATH]
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_edges
from src.data.run_config import load_model_run
from src.models.cascade import CascadeLP
from src.utils.log_utils import setup_logging
from src.utils.metrics import timer

INTERIM = 'data/interim/dsaa'
CHECKPOINTS = 'outputs/checkpoints/dsaa'
PREDICTIONS = 'outputs/predictions/dsaa'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default=PREDICTIONS,
                        help='Directory containing cascade_run_config.json')
    args = parser.parse_args()
    log = setup_logging('benchmark_inference')
    run = load_model_run(pathlib.Path(args.predictions), 'cascade')
    cfg = run['config'] if run else None
    raw = run['raw_path'] if run else 'data/raw/dsaa'
    interim = run['interim_path'] if run else INTERIM
    checkpoint = run['checkpoint'] if run else f'{CHECKPOINTS}/cascade.joblib'
    seed = cfg['seed'] if cfg else 42
    val_size = cfg['training']['val_size'] if cfg else 0.2
    train = load_edges(f'{raw}/train.csv')
    y = train['label'].values
    structural = pd.read_csv(f'{interim}/structural_train.csv', index_col='id')
    pos_features = np.load(f'{interim}/pos_train.npy')
    st_scores = pd.read_csv(f'{interim}/sentence_emb_train.csv', index_col='id')['st_score'].values

    not_self = (train['id1'] != train['id2']).values
    idx = np.where(not_self)[0]
    _, val = train_test_split(idx, test_size=val_size, stratify=y[idx], random_state=seed)

    val_pairs = train.iloc[val]
    val_structural = structural.iloc[val]
    val_pos = pos_features[val]
    val_st = st_scores[val]

    model = CascadeLP.load(checkpoint)

    log.info('Running single-pass predict() over %d validation pairs...', len(val_pairs))
    with timer() as t:
        y_pred, tier_used, _ = model.predict(val_structural, val_pos, val_st, val_pairs)
    wall_ms = t[0]
    ms_per_pair = wall_ms / len(val_pairs)
    rate_per_sec = len(val_pairs) / (wall_ms / 1000)

    log.info('Wall-clock  : %.2f s', wall_ms / 1000)
    log.info('Rate        : %.0f pairs/s', rate_per_sec)
    log.info('Latency     : %.4f ms/pair', ms_per_pair)

    out = {
        'n': len(val_pairs),
        'wall_clock_sec': wall_ms / 1000,
        'rate_per_sec': rate_per_sec,
        'ms_per_pair': ms_per_pair,
        'seed': seed,
        'val_size': val_size,
    }
    out_path = pathlib.Path(args.predictions) / 'throughput_benchmark.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info('Saved → %s', out_path)


if __name__ == '__main__':
    main()
