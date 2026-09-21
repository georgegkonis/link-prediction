"""
Show the id1-hub negative-sampling artifact (audit_negative_sampling.py) "in
action" against the thesis's own existing, unmodified CascadeLP validation
predictions (outputs/predictions/cascade_val_tiers.csv, from `make train
MODEL=cascade` on the original protocol) — no rerun, no protocol change.

Quantifies how much of the reported near-perfect Macro F1 is attributable to
a trivial 65-value id1 lookup versus how the same model does elsewhere.

Reads:
    data/raw/train.csv
    outputs/predictions/cascade_val_tiers.csv

Writes:
    outputs/stats/hub_in_predictions_audit.json

Usage:
    python -m scripts.analysis.audit_hub_in_predictions
"""
import argparse
import json
import pathlib

import pandas as pd
from sklearn.metrics import f1_score

from src.data.loader import load_edges
from src.utils.log_utils import setup_logging

log = setup_logging('audit_hub_in_predictions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='data/raw/train.csv')
    parser.add_argument('--tiers', default='outputs/predictions/cascade_val_tiers.csv')
    parser.add_argument('--hub-threshold', type=int, default=100)
    parser.add_argument('--output', default='outputs/stats/hub_in_predictions_audit.json')
    args = parser.parse_args()

    train = load_edges(args.train)
    df = train[train['id1'] != train['id2']]
    g1 = df.groupby('id1')['label'].agg(['mean', 'count'])
    hub_ids = set(g1[g1['count'] >= args.hub_threshold].index)
    log.info('Hub id1 (threshold=%d): %d', args.hub_threshold, len(hub_ids))

    tiers = pd.read_csv(args.tiers)  # already carries id1/id2 alongside y_true/y_pred/tier_used
    is_hub = tiers['id1'].isin(hub_ids)
    log.info('Validation pairs: %s, hub-sourced: %s (%.2f%%)',
             f'{len(tiers):,}', f'{is_hub.sum():,}', 100 * is_hub.mean())

    def macro_f1(sub):
        if len(sub) == 0 or sub['y_true'].nunique() < 2:
            return None
        return float(f1_score(sub['y_true'], sub['y_pred'], labels=[0, 1], average='macro', zero_division=0))

    overall_f1 = macro_f1(tiers)
    hub_f1 = macro_f1(tiers[is_hub])
    nonhub_f1 = macro_f1(tiers[~is_hub])

    # Trivial hub-lookup prediction on exactly the validation pairs actually evaluated.
    maj = (g1.loc[list(hub_ids), 'mean'] > 0.5).astype(int).to_dict()
    trivial_pred = tiers['id1'].map(maj).fillna(1).astype(int)
    trivial_f1 = float(f1_score(tiers['y_true'], trivial_pred, labels=[0, 1], average='macro', zero_division=0))
    agreement = float((trivial_pred == tiers['y_pred']).mean())
    hub_agreement = float((trivial_pred[is_hub] == tiers.loc[is_hub, 'y_pred']).mean()) if is_hub.sum() else None

    log.info('Overall CascadeLP Macro F1 on this validation set: %.6f', overall_f1)
    log.info('  restricted to hub-sourced pairs (%.2f%% of val):  %s',
             100 * is_hub.mean(), f'{hub_f1:.6f}' if hub_f1 is not None else 'undefined (single class, all negative)')
    log.info('  restricted to non-hub pairs:                     %s',
             f'{nonhub_f1:.6f}' if nonhub_f1 is not None else 'undefined (single class)')
    log.info('Trivial 65-value id1 lookup alone on this same validation set: Macro F1=%.6f', trivial_f1)
    log.info('Agreement between CascadeLP predictions and the trivial lookup: %.2f%% overall, '
             '%.2f%% on hub-sourced pairs', 100 * agreement, 100 * (hub_agreement or 0))

    report = {
        'hub_threshold': args.hub_threshold,
        'n_hub_ids': len(hub_ids),
        'n_val': len(tiers),
        'n_val_hub_sourced': int(is_hub.sum()),
        'val_hub_sourced_pct': float(100 * is_hub.mean()),
        'cascade_overall_macro_f1': overall_f1,
        'cascade_hub_subset_macro_f1': hub_f1,
        'cascade_nonhub_subset_macro_f1': nonhub_f1,
        'trivial_lookup_macro_f1_same_val_set': trivial_f1,
        'agreement_cascade_vs_trivial_lookup_pct': float(100 * agreement),
        'agreement_on_hub_subset_pct': float(100 * hub_agreement) if hub_agreement is not None else None,
    }
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info('Saved -> %s', out)


if __name__ == '__main__':
    main()
