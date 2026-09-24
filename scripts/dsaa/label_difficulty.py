"""
Compute separability metrics and difficulty categories.

Reads:
    data/raw/dsaa/{train,test}.csv
    data/interim/dsaa/{structural,tfidf}_{train,test}.csv
Writes:
    outputs/analyze_dataset-results.txt
    data/interim/dsaa/difficulty_train.csv
    data/interim/dsaa/difficulty_test.csv
    data/interim/dsaa/difficulty_thresholds.json

Usage:
    python -m scripts.dsaa.label_difficulty [--fpr 0.01]
"""

import argparse
import json
import pathlib

import pandas as pd

from src.data.loader import load_edges
from src.utils.difficulty import label_difficulty, pick_thresholds
from src.utils.log_utils import setup_logging

INTERIM = 'data/interim/dsaa'
OUTPUTS = 'outputs'


def main(cn_threshold: float | None, tfidf_threshold: float | None, fpr: float):
    pathlib.Path(OUTPUTS).mkdir(parents=True, exist_ok=True)
    pathlib.Path(INTERIM).mkdir(parents=True, exist_ok=True)
    log = setup_logging('label_difficulty')
    log.info('Loading edges and features...')
    train = load_edges('data/raw/dsaa/train.csv')
    test = load_edges('data/raw/dsaa/test.csv')
    y = train['label'].values

    cn_train = pd.read_csv(f'{INTERIM}/structural_train.csv', index_col='id')['cn'].values
    cn_test = pd.read_csv(f'{INTERIM}/structural_test.csv', index_col='id')['cn'].values
    tfidf_train = pd.read_csv(f'{INTERIM}/tfidf_train.csv', index_col='id')['tfidf_score'].values
    tfidf_test = pd.read_csv(f'{INTERIM}/tfidf_test.csv', index_col='id')['tfidf_score'].values

    not_self = (train['id1'] != train['id2']).values
    log.info('Common-neighbor distribution (non-self train pairs):\n  Positive : %s\n  Negative : %s',
             pd.Series(cn_train[not_self & (y == 1)]).describe(),
             pd.Series(cn_train[not_self & (y == 0)]).describe())

    log.info('TF-IDF cosine similarity distribution (non-self train pairs):\n  Positive : %s\n  Negative : %s',
             pd.Series(tfidf_train[not_self & (y == 1)]).describe(),
             pd.Series(tfidf_train[not_self & (y == 0)]).describe())

    if cn_threshold is None or tfidf_threshold is None:
        auto_cn, auto_tfidf = pick_thresholds(y, cn_train, tfidf_train, fpr=fpr)
        cn_threshold = cn_threshold if cn_threshold is not None else auto_cn
        tfidf_threshold = tfidf_threshold if tfidf_threshold is not None else auto_tfidf

    log.info('Trivial-pair thresholds: cn > %.3f  |  tfidf > %.3f  (fpr=%s)',
             cn_threshold, tfidf_threshold, fpr)

    diff_train = label_difficulty(train, cn_train, tfidf_train, cn_threshold, tfidf_threshold)
    diff_test = label_difficulty(test, cn_test, tfidf_test, cn_threshold, tfidf_threshold)

    train_counts = diff_train.value_counts()
    test_counts = diff_test.value_counts()

    lines = [
        'Separability Characterization',
        '==============================',
        '',
        f'Trivial-pair thresholds: cn > {cn_threshold:.3f}, tfidf > {tfidf_threshold:.3f} (fpr={fpr})',
        '',
        'Train difficulty breakdown:',
        *(f'  {k:22s}: {v:,} ({v / len(diff_train) * 100:.2f}%)' for k, v in train_counts.items()),
        '',
        'Test difficulty breakdown:',
        *(f'  {k:22s}: {v:,} ({v / len(diff_test) * 100:.2f}%)' for k, v in test_counts.items()),
    ]
    report = '\n'.join(lines)
    log.info('\n%s', report)

    diff_train.to_csv(f'{INTERIM}/difficulty_train.csv')
    diff_test.to_csv(f'{INTERIM}/difficulty_test.csv')
    with open(f'{OUTPUTS}/analyze_dataset-results.txt', 'w') as f:
        f.write(report + '\n')
    pathlib.Path(f'{INTERIM}/difficulty_thresholds.json').write_text(
        json.dumps({'cn_threshold': float(cn_threshold), 'tfidf_threshold': float(tfidf_threshold)}, indent=2)
    )

    log.info('Saved → %s/difficulty_{train,test}.csv', INTERIM)
    log.info('Saved → %s/difficulty_thresholds.json', INTERIM)
    log.info('Saved → %s/analyze_dataset-results.txt', OUTPUTS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cn-threshold', type=float, default=None)
    parser.add_argument('--tfidf-threshold', type=float, default=None)
    parser.add_argument('--fpr', type=float, default=0.01)
    args = parser.parse_args()
    main(args.cn_threshold, args.tfidf_threshold, args.fpr)
