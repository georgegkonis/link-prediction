"""
Threshold ablation for CascadeLP (thesis Ch.5 SS5.1): sweep tau1/tau2 on the
already-trained checkpoint (routing thresholds don't affect tier fitting, only
.predict(), so no retraining is needed) and report overall Macro F1 and
Tier-3 call rate for each combination.

Usage:
    python -m scripts.ablate_cascade_thresholds
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.data.loader import load_edges
from src.models.cascade import CascadeLP
from src.utils.log_utils import setup_logging

INTERIM = 'data/interim'
CHECKPOINTS = 'outputs/checkpoints'

T1_GRID = [0.6, 0.7, 0.8, 0.9, 0.95]
T2_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    log = setup_logging('ablate_cascade_thresholds')
    train = load_edges('data/raw/train.csv')
    y = train['label'].values
    structural = pd.read_csv(f'{INTERIM}/structural_train.csv', index_col='id')
    pos_features = np.load(f'{INTERIM}/pos_train.npy')
    st_scores = pd.read_csv(f'{INTERIM}/sentence_emb_train.csv', index_col='id')['st_score'].values

    not_self = (train['id1'] != train['id2']).values
    idx = np.where(not_self)[0]
    _, val = train_test_split(idx, test_size=0.2, stratify=y[idx], random_state=42)

    val_pairs = train.iloc[val]
    val_structural = structural.iloc[val]
    val_pos = pos_features[val]
    val_st = st_scores[val]
    y_val = y[val]

    model = CascadeLP.load(f'{CHECKPOINTS}/cascade.joblib')

    log.info('%6s %6s %10s %9s %9s %9s', 'tau1', 'tau2', 'macro_f1', 'tier1_%', 'tier2_%', 'tier3_%')
    rows = []
    for t1 in T1_GRID:
        for t2 in T2_GRID:
            model.tier1_threshold = t1
            model.tier2_threshold = t2
            y_pred, tier_used = model.predict(val_structural, val_pos, val_st, val_pairs)
            mf1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
            pct = lambda t: 100 * (tier_used == t).mean()
            row = {'tau1': t1, 'tau2': t2, 'macro_f1': mf1,
                   'tier1_pct': pct(1), 'tier2_pct': pct(2), 'tier3_pct': pct(3)}
            rows.append(row)
            log.info('%6.2f %6.2f %10.4f %9.2f %9.2f %9.2f', t1, t2, mf1, pct(1), pct(2), pct(3))

    pd.DataFrame(rows).to_csv('outputs/predictions/cascade_threshold_ablation.csv', index=False)
    log.info('Saved → outputs/predictions/cascade_threshold_ablation.csv')


if __name__ == '__main__':
    main()
