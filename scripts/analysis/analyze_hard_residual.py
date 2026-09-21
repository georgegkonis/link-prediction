"""
Programmatic analysis of CascadeLP's "hard residual" — the pairs that reach
Tier 3 (EmbeddingClassifier) and are still labelled 'hard' (thesis Ch.5 §5.2).
Replaces the previously hand-transcribed constants in generate_macros.py with
a real join against nodes.tsv, checking whether missing/empty node text
explains Tier 3's collapse on this subset.

Outputs:
    outputs/predictions/dsaa/hard_residual_analysis.json

Usage:
    python -m scripts.analysis.analyze_hard_residual
"""

import json
import pathlib

import pandas as pd

from src.data.loader import load_nodes_for_ids
from src.utils.log_utils import setup_logging

RAW = 'data/raw/dsaa'
PREDICTIONS = 'outputs/predictions/dsaa'


def main():
    log = setup_logging('analyze_hard_residual')
    vt = pd.read_csv(f'{PREDICTIONS}/cascade_val_tiers.csv')
    hard_res = vt[(vt['tier_used'] == 3) & (vt['difficulty'] == 'hard')].copy()
    n_hr = len(hard_res)
    log.info('Hard residual: %d pairs', n_hr)

    node_ids = set(hard_res['id1'].tolist()) | set(hard_res['id2'].tolist())
    log.info('Streaming nodes.tsv for %d node ids...', len(node_ids))
    nodes = load_nodes_for_ids(f'{RAW}/nodes.tsv', node_ids)

    def has_text(node_id) -> bool:
        if node_id not in nodes.index:
            return False
        text = nodes.loc[node_id, 'text']
        return isinstance(text, str) and len(text.strip()) > 0

    hard_res['id1_has_text'] = hard_res['id1'].map(has_text)
    hard_res['id2_has_text'] = hard_res['id2'].map(has_text)
    hard_res['complete'] = hard_res['id1_has_text'] & hard_res['id2_has_text']

    complete = hard_res[hard_res['complete']]
    incomplete = hard_res[~hard_res['complete']]

    n_missing = len(incomplete)
    n_complete = len(complete)
    missing_pct = 100 * n_missing / n_hr if n_hr else float('nan')
    complete_acc = complete['correct'].mean() if len(complete) else float('nan')
    incomplete_acc = incomplete['correct'].mean() if len(incomplete) else float('nan')

    log.info('Missing-text pairs : %d (%.1f%%)', n_missing, missing_pct)
    log.info('Complete-text pairs: %d, accuracy=%.4f', n_complete, complete_acc)
    log.info('Missing-text pairs : accuracy=%.4f', incomplete_acc)

    # Node appearing in the most hard-residual pairs (endpoint on either side)
    node_counts = pd.concat([hard_res['id1'], hard_res['id2']]).value_counts()
    top_node_id, top_node_pairs = None, 0
    top_node_pred_pos_pct = float('nan')
    if len(node_counts):
        top_node_id = int(node_counts.index[0])
        top_node_pairs = int(node_counts.iloc[0])
        involved = hard_res[(hard_res['id1'] == top_node_id) | (hard_res['id2'] == top_node_id)]
        top_node_pred_pos_pct = 100 * involved['y_pred'].mean()
        log.info('Top node in hard residual: id=%d, pairs=%d, pred_pos_pct=%.1f%%',
                  top_node_id, top_node_pairs, top_node_pred_pos_pct)

    out = {
        'n': n_hr,
        'missing_n': n_missing,
        'missing_pct': missing_pct,
        'complete_n': n_complete,
        'complete_acc': complete_acc,
        'incomplete_acc': incomplete_acc,
        'top_node_id': top_node_id,
        'top_node_pairs': top_node_pairs,
        'top_node_pred_pos_pct': top_node_pred_pos_pct,
    }
    out_path = pathlib.Path(PREDICTIONS) / 'hard_residual_analysis.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info('Saved → %s', out_path)


if __name__ == '__main__':
    main()
