"""
Audit train/test data leakage.

Writes:
    outputs/leakage_audit-results.txt
    data/interim/dsaa/leakage_pairs.csv

Usage:
    python -m scripts.dsaa.audit_pairs
"""

import argparse

import pandas as pd

from src.data.loader import load_edges
from src.utils.log_utils import setup_logging

OUTPUTS = 'outputs'
INTERIM = 'data/interim/dsaa'


def _undirected_key(df: pd.DataFrame) -> pd.Series:
    lo = df[['id1', 'id2']].min(axis=1)
    hi = df[['id1', 'id2']].max(axis=1)
    return lo.astype(str) + '_' + hi.astype(str)


def pair_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Exact and reversed (u, v) overlap between train and test edge lists."""
    train_pairs = set(zip(train['id1'], train['id2']))
    test_pairs = list(zip(test['id1'], test['id2']))

    exact = [p for p in test_pairs if p in train_pairs]
    reversed_ = [p for p in test_pairs if (p[1], p[0]) in train_pairs and p not in train_pairs]

    return {
        'exact_overlap': exact,
        'reversed_overlap': reversed_,
        'exact_count': len(exact),
        'reversed_count': len(reversed_),
        'test_count': len(test_pairs),
    }


def intra_train_duplicates(train: pd.DataFrame) -> pd.DataFrame:
    """Undirected duplicate pairs within train.csv; flags label conflicts if any."""
    keyed = train.assign(_key=_undirected_key(train))
    dupes = keyed[keyed.duplicated('_key', keep=False)].sort_values('_key')
    conflicts = dupes.groupby('_key')['label'].nunique()
    dupes = dupes.assign(label_conflict=dupes['_key'].map(conflicts > 1))
    return dupes


def self_loop_report(df: pd.DataFrame, name: str) -> dict:
    """Count + label distribution (if labels present) of id1==id2 rows."""
    loops = df[df['id1'] == df['id2']]
    report = {'name': name, 'total': len(df), 'self_loops': len(loops)}
    if 'label' in df.columns:
        report['label_counts'] = loops['label'].value_counts().to_dict()
    return report


def main(train_path: str, test_path: str):
    log = setup_logging('audit_leakage')
    log.info('Loading edges...')
    train = load_edges(train_path)
    test = load_edges(test_path)

    log.info('Self-loop report...')
    train_loops = self_loop_report(train, 'train')
    test_loops = self_loop_report(test, 'test')

    log.info('Checking train/test pair overlap...')
    overlap = pair_overlap(train, test)

    log.info('Checking intra-train duplicate pairs...')
    dupes = intra_train_duplicates(train)

    lines = [
        'Leakage Audit',
        '=============',
        '',
        f"Train self-loops : {train_loops['self_loops']:,} / {train_loops['total']:,}"
        f"  labels={train_loops.get('label_counts')}",
        f"Test  self-loops : {test_loops['self_loops']:,} / {test_loops['total']:,}",
        '',
        f"Exact (id1,id2) overlap train∩test    : {overlap['exact_count']:,}"
        f" ({overlap['exact_count'] / overlap['test_count'] * 100:.4f}% of test)",
        f"Reversed (id2,id1) overlap train∩test : {overlap['reversed_count']:,}"
        f" ({overlap['reversed_count'] / overlap['test_count'] * 100:.4f}% of test)",
        '',
        f"Intra-train duplicate undirected pairs : {dupes['_key'].nunique():,} groups"
        f"  ({len(dupes):,} rows), label_conflicts={int(dupes['label_conflict'].sum()) if len(dupes) else 0}",
    ]
    report = '\n'.join(lines)
    log.info('\n%s', report)

    exact_set = set(overlap['exact_overlap'])
    reversed_set = set(overlap['reversed_overlap'])
    exact_mask = test.apply(lambda r: (r['id1'], r['id2']) in exact_set, axis=1)
    reversed_mask = test.apply(lambda r: (r['id1'], r['id2']) in reversed_set, axis=1)
    overlap_rows = pd.concat([
        test.loc[exact_mask].assign(kind='exact'),
        test.loc[reversed_mask].assign(kind='reversed'),
    ])

    overlap_rows.to_csv(f'{INTERIM}/leakage_pairs.csv')
    with open(f'{OUTPUTS}/leakage_audit-results.txt', 'w') as f:
        f.write(report + '\n')

    log.info('Saved → %s/leakage_pairs.csv', INTERIM)
    log.info('Saved → %s/leakage_audit-results.txt', OUTPUTS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/raw/dsaa/train.csv')
    parser.add_argument('--test', default='data/raw/dsaa/test.csv')
    args = parser.parse_args()
    main(args.train, args.test)
