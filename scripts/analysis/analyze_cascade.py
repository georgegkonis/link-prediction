"""
Post-hoc analysis of CascadeLP validation results for the thesis (Ch.4 §4.4, Ch.5).

Reads outputs/predictions/dsaa/cascade_val_tiers.csv (written by scripts/train.py) plus
the structural features, and prints:
  - overall macro-F1 / accuracy on the validation split
  - per-tier n / call-rate / accuracy / macro-F1
  - per-difficulty n / accuracy / macro-F1
  - tier x difficulty cross-tab
  - cold-start (zero common neighbours) per-tier accuracy / macro-F1
  - the "hard residual" reaching Tier 3 and still labelled 'hard'
  - a small sample of Tier-3 hard misclassifications

Usage:
    python -m scripts.analyze_cascade
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.utils.log_utils import setup_logging

PREDICTIONS = 'outputs/predictions/dsaa'
INTERIM = 'data/interim/dsaa'

TIER_NAMES = {0: 'Tier0 self-loop', 1: 'Tier1 structural',
              2: 'Tier2 POS', 3: 'Tier3 embedding'}


def _mf1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def main():
    log = setup_logging('analyze_cascade')
    df = pd.read_csv(f'{PREDICTIONS}/cascade_val_tiers.csv')
    n = len(df)
    yt, yp = df['y_true'].values, df['y_pred'].values

    log.info('Validation pairs: %s', f'{n:,}')
    log.info('Overall accuracy : %.4f', accuracy_score(yt, yp))
    log.info('Overall macro-F1 : %.4f', _mf1(yt, yp))

    # Per-tier
    rows = []
    for t in sorted(df['tier_used'].unique()):
        m = df['tier_used'] == t
        rows.append({
            'tier': TIER_NAMES.get(t, t), 'n': int(m.sum()),
            'call_rate_%': round(100 * m.mean(), 2),
            'accuracy': round(accuracy_score(yt[m], yp[m]), 4),
            'macro_f1': round(_mf1(yt[m], yp[m]), 4),
        })
    log.info('=== Per-tier ===\n%s', pd.DataFrame(rows).to_string(index=False))

    # Per-difficulty
    rows = []
    for d in df['difficulty'].unique():
        m = df['difficulty'] == d
        rows.append({
            'difficulty': d, 'n': int(m.sum()),
            'share_%': round(100 * m.mean(), 2),
            'accuracy': round(accuracy_score(yt[m], yp[m]), 4),
            'macro_f1': round(_mf1(yt[m], yp[m]), 4),
        })
    log.info('=== Per-difficulty ===\n%s', pd.DataFrame(rows).to_string(index=False))

    # Tier x difficulty
    piv_n = df.pivot_table(index='tier_used', columns='difficulty',
                           values='y_true', aggfunc='count', fill_value=0)
    df['_correct'] = (yt == yp).astype(int)
    piv_acc = df.pivot_table(index='tier_used', columns='difficulty',
                             values='_correct', aggfunc='mean')
    log.info('=== Tier x difficulty ===\nn:\n%s\n\naccuracy:\n%s',
             piv_n.to_string(), piv_acc.round(4).to_string())

    # Cold-start (zero common neighbours) — join CN by id from structural_train
    cn = pd.read_csv(f'{INTERIM}/structural_train.csv', index_col='id')['cn']
    df = df.merge(cn.rename('cn'), left_on='id', right_index=True, how='left')
    cs = (df['id1'] != df['id2']) & (df['cn'].fillna(0) == 0)
    log.info('=== Cold-start subset (CN==0, non-self-loop): %s pairs ===', f'{int(cs.sum()):,}')
    log.info('Cold-start accuracy : %.4f', accuracy_score(yt[cs.values], yp[cs.values]))
    log.info('Cold-start macro-F1 : %.4f', _mf1(yt[cs.values], yp[cs.values]))
    rows = []
    sub = df[cs]
    for t in sorted(sub['tier_used'].unique()):
        m = sub['tier_used'] == t
        rows.append({
            'tier': TIER_NAMES.get(t, t), 'n': int(m.sum()),
            'share_of_cs_%': round(100 * m.mean(), 2),
            'accuracy': round(accuracy_score(sub['y_true'][m], sub['y_pred'][m]), 4),
            'macro_f1': round(_mf1(sub['y_true'][m], sub['y_pred'][m]), 4),
        })
    log.info('%s', pd.DataFrame(rows).to_string(index=False))

    # Hard residual reaching Tier 3
    hard_t3 = (df['tier_used'] == 3) & (df['difficulty'] == 'hard')
    log.info('=== Hard residual at Tier 3: %s pairs (%.2f%% of val) ===',
             f'{int(hard_t3.sum()):,}', 100 * hard_t3.mean())
    if hard_t3.sum():
        log.info('  accuracy : %.4f', accuracy_score(yt[hard_t3.values], yp[hard_t3.values]))
        log.info('  macro-F1 : %.4f', _mf1(yt[hard_t3.values], yp[hard_t3.values]))

    # Error concentration by difficulty
    err = df[yt != yp]
    conc = (err['difficulty'].value_counts(normalize=True) * 100).round(2)
    log.info('=== Error concentration by difficulty ===\nTotal errors: %s\n%s',
             f'{len(err):,}', conc.to_string())


if __name__ == '__main__':
    main()
