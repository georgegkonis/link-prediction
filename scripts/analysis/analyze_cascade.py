"""
Post-hoc analysis of CascadeLP validation results for the thesis (Ch.4 §4.4, Ch.5).

Reads outputs/predictions/cascade_val_tiers.csv (written by scripts/train.py) plus
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

PREDICTIONS = 'outputs/predictions'
INTERIM = 'data/interim'

TIER_NAMES = {0: 'Tier0 self-loop', 1: 'Tier1 structural',
              2: 'Tier2 POS', 3: 'Tier3 embedding'}


def _mf1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def main():
    df = pd.read_csv(f'{PREDICTIONS}/cascade_val_tiers.csv')
    n = len(df)
    yt, yp = df['y_true'].values, df['y_pred'].values

    print(f'Validation pairs: {n:,}')
    print(f'Overall accuracy : {accuracy_score(yt, yp):.4f}')
    print(f'Overall macro-F1 : {_mf1(yt, yp):.4f}\n')

    # Per-tier
    print('=== Per-tier ===')
    rows = []
    for t in sorted(df['tier_used'].unique()):
        m = df['tier_used'] == t
        rows.append({
            'tier': TIER_NAMES.get(t, t), 'n': int(m.sum()),
            'call_rate_%': round(100 * m.mean(), 2),
            'accuracy': round(accuracy_score(yt[m], yp[m]), 4),
            'macro_f1': round(_mf1(yt[m], yp[m]), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Per-difficulty
    print('\n=== Per-difficulty ===')
    rows = []
    for d in df['difficulty'].unique():
        m = df['difficulty'] == d
        rows.append({
            'difficulty': d, 'n': int(m.sum()),
            'share_%': round(100 * m.mean(), 2),
            'accuracy': round(accuracy_score(yt[m], yp[m]), 4),
            'macro_f1': round(_mf1(yt[m], yp[m]), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Tier x difficulty
    print('\n=== Tier x difficulty (n / accuracy) ===')
    piv_n = df.pivot_table(index='tier_used', columns='difficulty',
                           values='y_true', aggfunc='count', fill_value=0)
    print('n:\n', piv_n.to_string())
    df['_correct'] = (yt == yp).astype(int)
    piv_acc = df.pivot_table(index='tier_used', columns='difficulty',
                             values='_correct', aggfunc='mean')
    print('\naccuracy:\n', piv_acc.round(4).to_string())

    # Cold-start (zero common neighbours) — join CN by id from structural_train
    cn = pd.read_csv(f'{INTERIM}/structural_train.csv', index_col='id')['cn']
    df = df.merge(cn.rename('cn'), left_on='id', right_index=True, how='left')
    # cold-start: not a self-loop, CN is 0 or NaN (node absent / no shared nbr)
    cs = (df['id1'] != df['id2']) & (df['cn'].fillna(0) == 0)
    print(f'\n=== Cold-start subset (CN==0, non-self-loop): {int(cs.sum()):,} pairs ===')
    print(f'Cold-start accuracy : {accuracy_score(yt[cs.values], yp[cs.values]):.4f}')
    print(f'Cold-start macro-F1 : {_mf1(yt[cs.values], yp[cs.values]):.4f}')
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
    print(pd.DataFrame(rows).to_string(index=False))

    # Hard residual reaching Tier 3
    hard_t3 = (df['tier_used'] == 3) & (df['difficulty'] == 'hard')
    print(f'\n=== Hard residual at Tier 3: {int(hard_t3.sum()):,} pairs '
          f'({100 * hard_t3.mean():.2f}% of val) ===')
    if hard_t3.sum():
        print(f'  accuracy : {accuracy_score(yt[hard_t3.values], yp[hard_t3.values]):.4f}')
        print(f'  macro-F1 : {_mf1(yt[hard_t3.values], yp[hard_t3.values]):.4f}')

    # Error concentration by difficulty
    print('\n=== Error concentration by difficulty ===')
    err = df[yt != yp]
    print(f'Total errors: {len(err):,}')
    conc = (err['difficulty'].value_counts(normalize=True) * 100).round(2)
    print(conc.to_string())


if __name__ == '__main__':
    main()
