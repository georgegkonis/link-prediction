"""
Separability characterization: how much of DSAA 2023 is actually "hard"?

For train (labeled) pairs, computes common-neighbor and TF-IDF cosine
similarity distributions for positive vs. negative pairs, picks data-driven
"trivial pair" thresholds, and reports what fraction of train/test falls
into each difficulty bucket.

Outputs:
    outputs/analyze_dataset-results.txt
    data/interim/difficulty_train.csv   (id, difficulty)
    data/interim/difficulty_test.csv    (id, difficulty)

Usage:
    python -m scripts.analyze_dataset [--cn-threshold F] [--tfidf-threshold F] [--fpr F]
"""

import argparse

import pandas as pd

from src.data.loader import load_edges
from src.utils.difficulty import label_difficulty, pick_thresholds

INTERIM = 'data/interim'
OUTPUTS = 'outputs'


def main(cn_threshold: float | None, tfidf_threshold: float | None, fpr: float):
    print('Loading edges and features...')
    train = load_edges('data/raw/train.csv')
    test = load_edges('data/raw/test.csv')
    y = train['label'].values

    cn_train = pd.read_csv(f'{INTERIM}/structural_train.csv', index_col='id')['cn'].values
    cn_test = pd.read_csv(f'{INTERIM}/structural_test.csv', index_col='id')['cn'].values
    tfidf_train = pd.read_csv(f'{INTERIM}/tfidf_train.csv', index_col='id')['tfidf_score'].values
    tfidf_test = pd.read_csv(f'{INTERIM}/tfidf_test.csv', index_col='id')['tfidf_score'].values

    not_self = (train['id1'] != train['id2']).values
    print('\nCommon-neighbor distribution (non-self train pairs):')
    print(f'  Positive : {pd.Series(cn_train[not_self & (y == 1)]).describe()}')
    print(f'  Negative : {pd.Series(cn_train[not_self & (y == 0)]).describe()}')

    print('\nTF-IDF cosine similarity distribution (non-self train pairs):')
    print(f'  Positive : {pd.Series(tfidf_train[not_self & (y == 1)]).describe()}')
    print(f'  Negative : {pd.Series(tfidf_train[not_self & (y == 0)]).describe()}')

    if cn_threshold is None or tfidf_threshold is None:
        auto_cn, auto_tfidf = pick_thresholds(y, cn_train, tfidf_train, fpr=fpr)
        cn_threshold = cn_threshold if cn_threshold is not None else auto_cn
        tfidf_threshold = tfidf_threshold if tfidf_threshold is not None else auto_tfidf

    print(f'\nTrivial-pair thresholds: cn > {cn_threshold:.3f}  |  tfidf > {tfidf_threshold:.3f}'
          f'  (fpr={fpr})')

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
    print('\n' + report)

    diff_train.to_csv(f'{INTERIM}/difficulty_train.csv')
    diff_test.to_csv(f'{INTERIM}/difficulty_test.csv')
    with open(f'{OUTPUTS}/analyze_dataset-results.txt', 'w') as f:
        f.write(report + '\n')

    print(f'\nSaved → {INTERIM}/difficulty_{{train,test}}.csv')
    print(f'Saved → {OUTPUTS}/analyze_dataset-results.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cn-threshold', type=float, default=None)
    parser.add_argument('--tfidf-threshold', type=float, default=None)
    parser.add_argument('--fpr', type=float, default=0.01)
    args = parser.parse_args()
    main(args.cn_threshold, args.tfidf_threshold, args.fpr)
