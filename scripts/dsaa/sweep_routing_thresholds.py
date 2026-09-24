"""
Ablate cascade thresholds to measure performance changes.

Usage:
    python -m scripts.dsaa.sweep_routing_thresholds
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.data.loader import load_edges
from src.models.cascade import CascadeLP, cold_start_from_structural
from src.utils.log_utils import setup_logging

INTERIM = 'data/interim/dsaa'
CHECKPOINTS = 'outputs/checkpoints/dsaa'
PREDICTIONS = 'outputs/predictions/dsaa'

T1_GRID = [0.6, 0.7, 0.8, 0.9, 0.95]
T2_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]


def _tier_f1(y_val, y_pred, tier_used, tier_id) -> float:
    mask = tier_used == tier_id
    if not mask.any():
        return float('nan')
    return f1_score(y_val[mask], y_pred[mask], average='macro', zero_division=0)


def main():
    log = setup_logging('ablate_cascade_thresholds')
    train = load_edges('data/raw/dsaa/train.csv')
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

    # ---- Tier 2 confidence ceiling: how many pairs even reach Tier 2, and of
    # those, how many get a unanimous (proba == 1.0) RandomForest vote that no
    # tau2 short of >1.0 could ever escalate past?
    model.tier1_threshold = 0.8  # deployed default; irrelevant to the Tier-2 pool composition below
    cold_start = cold_start_from_structural(val_structural)
    not_cold = ~cold_start
    proba1 = model.tier1.predict_proba(val_structural[not_cold])
    conf1 = proba1.max(axis=1)
    tier2_eligible = np.ones(len(val_pairs), dtype=bool)
    tier2_eligible[np.where(not_cold)[0][conf1 >= model.tier1_threshold]] = False
    proba2_all = model.tier2.predict_proba(val_pos[tier2_eligible])
    conf2 = proba2_all.max(axis=1)
    tier2_saturation = {
        'n_tier2_eligible': int(tier2_eligible.sum()),
        'pct_conf_ge_0.99': float(100 * (conf2 >= 0.99).mean()),
        'pct_conf_ge_0.999': float(100 * (conf2 >= 0.999).mean()),
        'pct_conf_eq_1.0': float(100 * (conf2 >= 1.0).mean()),
        'n_estimators': int(model.tier2.clf.n_estimators),
    }
    log.info('Tier-2 confidence ceiling: %s', tier2_saturation)
    with open(f'{PREDICTIONS}/tier2_confidence_saturation.json', 'w') as f:
        json.dump(tier2_saturation, f, indent=2)

    log.info('%6s %7s %10s %9s %9s %9s  %9s %9s %9s', 'tau1', 'tau2', 'macro_f1',
              'tier1_%', 'tier2_%', 'tier3_%', 'tier1_f1', 'tier2_f1', 'tier3_f1')
    rows = []
    for t1 in T1_GRID:
        for t2 in T2_GRID:
            model.tier1_threshold = t1
            model.tier2_threshold = t2
            y_pred, tier_used, _ = model.predict(val_structural, val_pos, val_st, val_pairs)
            mf1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
            pct = lambda t: 100 * (tier_used == t).mean()
            row = {'tau1': t1, 'tau2': t2, 'macro_f1': mf1,
                   'tier1_pct': pct(1), 'tier2_pct': pct(2), 'tier3_pct': pct(3),
                   'tier1_n': int((tier_used == 1).sum()), 'tier2_n': int((tier_used == 2).sum()),
                   'tier3_n': int((tier_used == 3).sum()),
                   'tier1_f1': _tier_f1(y_val, y_pred, tier_used, 1),
                   'tier2_f1': _tier_f1(y_val, y_pred, tier_used, 2),
                   'tier3_f1': _tier_f1(y_val, y_pred, tier_used, 3)}
            rows.append(row)
            log.info('%6.2f %7.4f %10.4f %9.2f %9.2f %9.2f  %9.4f %9.4f %9.4f',
                      t1, t2, mf1, pct(1), pct(2), pct(3), row['tier1_f1'], row['tier2_f1'], row['tier3_f1'])

    pd.DataFrame(rows).to_csv(f'{PREDICTIONS}/cascade_threshold_ablation.csv', index=False)
    log.info('Saved → %s/cascade_threshold_ablation.csv', PREDICTIONS)


if __name__ == '__main__':
    main()
