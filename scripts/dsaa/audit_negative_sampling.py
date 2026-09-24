"""
Audit the negative sampling distribution in the dataset.

Reads:
    data/raw/dsaa/{train,test}.csv by default
Writes:
    outputs/stats/negative_sampling_audit.json

Usage:
    python -m scripts.dsaa.audit_negative_sampling [--train PATH] [--test PATH] [--output PATH]
"""
import argparse
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import load_edges
from src.utils.log_utils import setup_logging

log = setup_logging('audit_negative_sampling')


def hub_stats(df: pd.DataFrame, threshold: int) -> dict:
    g1 = df.groupby('id1')['label'].agg(['mean', 'count'])
    hubs = g1[g1['count'] >= threshold]
    hub_ids = set(hubs.index)
    pure = ((hubs['mean'] < 0.01) | (hubs['mean'] > 0.99))

    maj = (hubs['mean'] > 0.5).astype(int).to_dict()
    pred = df['id1'].map(maj).fillna(1).astype(int)  # non-hub rows: predict positive
    acc = accuracy_score(df['label'], pred)
    f1 = f1_score(df['label'], pred, average='macro')

    non_hub = df[~df['id1'].isin(hub_ids)]
    hub_rows = df[df['id1'].isin(hub_ids)]

    return {
        'threshold': threshold,
        'n_hub_id1': int(len(hubs)),
        'n_unique_id1': int(df['id1'].nunique()),
        'hub_row_count': int(hubs['count'].sum()),
        'hub_row_pct': float(100 * hubs['count'].sum() / len(df)),
        'hub_purity_fraction': float(pure.mean()),
        'non_hub_row_count': int(len(non_hub)),
        'non_hub_positive_rate': float(non_hub['label'].mean()) if len(non_hub) else None,
        'hub_negative_rate': float(1 - hub_rows['label'].mean()) if len(hub_rows) else None,
        'trivial_lookup_accuracy': float(acc),
        'trivial_lookup_macro_f1': float(f1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='data/raw/dsaa/train.csv')
    parser.add_argument('--test', default='data/raw/dsaa/test.csv')
    parser.add_argument('--output', default='outputs/stats/negative_sampling_audit.json')
    args = parser.parse_args()

    train = load_edges(args.train)
    test = load_edges(args.test)
    df = train[train['id1'] != train['id2']].copy()
    log.info('Non-self-loop train rows: %s', f'{len(df):,}')

    thresholds = [50, 100, 200]
    results = [hub_stats(df, t) for t in thresholds]
    for r in results:
        log.info('threshold=%d: %d hub id1 covering %.2f%% of rows (purity=%.4f), '
                 'trivial-lookup Macro F1=%.6f',
                 r['threshold'], r['n_hub_id1'], r['hub_row_pct'], r['hub_purity_fraction'],
                 r['trivial_lookup_macro_f1'])

    # Fix the reference threshold (100) for the headline numbers + test-set overlap check.
    g1 = df.groupby('id1')['label'].agg(['mean', 'count'])
    hubs = g1[g1['count'] >= 100]
    hub_ids = set(hubs.index)
    test_hub_overlap = len(hub_ids & set(test['id1'].unique()))
    test_hub_row_pct = float(100 * test['id1'].isin(hub_ids).mean())
    log.info('Hub id1 (threshold=100) found in test.csv: %d / %d (%.2f%% of test rows carry a hub id1)',
             test_hub_overlap, len(hub_ids), test_hub_row_pct)

    # id2-side comparison, for completeness — is there an analogous (weaker) artifact there?
    g2 = df.groupby('id2')['label'].agg(['mean', 'count'])
    hubs2 = g2[g2['count'] >= 100]
    pure2 = ((hubs2['mean'] < 0.01) | (hubs2['mean'] > 0.99))
    log.info('id2-side: %d hub id2 (threshold=100), purity=%.4f, row coverage=%.2f%%',
             len(hubs2), pure2.mean() if len(hubs2) else float('nan'),
             100 * hubs2['count'].sum() / len(df) if len(hubs2) else 0.0)

    report = {
        'n_train_rows_no_self_loop': int(len(df)),
        'overall_positive_rate': float(df['label'].mean()),
        'by_threshold': results,
        'reference_threshold': 100,
        'reference_hub_ids': sorted(int(i) for i in hub_ids),
        'test_hub_id1_overlap': test_hub_overlap,
        'test_rows_with_hub_id1_pct': test_hub_row_pct,
        'id2_side': {
            'n_hub_id2': int(len(hubs2)),
            'purity_fraction': float(pure2.mean()) if len(hubs2) else None,
            'row_coverage_pct': float(100 * hubs2['count'].sum() / len(df)) if len(hubs2) else 0.0,
        },
    }
    import pathlib
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info('Saved -> %s', out_path)


if __name__ == '__main__':
    main()
