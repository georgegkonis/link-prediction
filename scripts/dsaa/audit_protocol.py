"""
Cached-data protocol audit for the DSAA 2023 legacy row-split/full-positive-graph
protocol used by ``make train``/``make predict-test`` (Chapter 4, S:data-splits): how many
validation positives already contributed edges to the structural graph before the
split, and (with --swap) how sensitive the trained CascadeLP checkpoint is to
endpoint order. Reads only cached features/checkpoints already produced by
``make run SCRIPT=dsaa.compute_structural``/``make train MODEL=cascade``; performs no feature
extraction and no retraining.

Adapted from the earlier graph-holdout revision for this branch's cached DSAA
features and CascadeLP.predict() signature.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_edges
from src.data.protocol import fingerprint, pair_keys


def swap_pos(features):
    if features.ndim != 2 or features.shape[1] % 2:
        raise ValueError('Expected two equally sized endpoint blocks')
    half = features.shape[1] // 2
    return np.concatenate([features[:, half:], features[:, :half]], axis=1)


def swap_summary(original, reversed_, probability, reversed_probability):
    delta = np.abs(probability - reversed_probability)
    return {'n': len(original), 'label_changes': int((original != reversed_).sum()),
            'label_change_pct': float(100 * (original != reversed_).mean()),
            'mean_abs_probability_change': float(delta.mean()),
            'probability_change_quantiles': dict(zip(['0', '50', '90', '99', '100'],
                                                       np.quantile(delta, [0, .5, .9, .99, 1]).tolist()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='data/raw/dsaa')
    parser.add_argument('--interim', default='data/interim/dsaa')
    parser.add_argument('--checkpoints', default='outputs/checkpoints/dsaa')
    parser.add_argument('--predictions', default='outputs/predictions/dsaa')
    parser.add_argument('--val-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='outputs/stats/supervisor_audit.json')
    parser.add_argument('--swap', action='store_true', help='Run cached model inference; never extracts features')
    parser.add_argument('--limit', type=int, default=None, help='Limit swap inference only; default full validation')
    parser.add_argument('--pairs-output', default=None, help='Optional CSV of original/reversed predictions')
    args = parser.parse_args()

    pairs = load_edges(f'{args.raw}/train.csv')
    h = pd.read_csv(f'{args.interim}/structural_train.csv', index_col='id')
    if not pairs.index.equals(h.index):
        raise ValueError('Structural cache ID mismatch')
    idx = np.flatnonzero((pairs.id1 != pairs.id2).to_numpy())
    tr, val = train_test_split(idx, test_size=args.val_size, stratify=pairs.label.to_numpy()[idx],
                                random_state=args.seed)
    vp = pairs.iloc[val]
    report = {'protocol': 'legacy_row_split_full_positive_graph', 'pairs_sha256': fingerprint(pairs),
              'n_train': len(tr), 'n_val': len(val), 'validation_positives_in_graph': int(vp.label.sum()),
              'cross_split_unordered_groups': len(pair_keys(pairs.iloc[tr]).intersection(pair_keys(vp))),
              'zero_cn_count': int((h.iloc[val].cn == 0).sum())}
    for name, source in [('full_graph', pairs), ('training_partition_graph', pairs.iloc[tr])]:
        positive = source[source.label == 1]
        nodes = set(positive[['id1', 'id2']].to_numpy().ravel())
        a, b = ~vp.id1.isin(nodes), ~vp.id2.isin(nodes)
        report[name] = {'one_unobserved': int((a ^ b).sum()), 'both_unobserved': int((a & b).sum()),
                         'cold_positive': int(vp.loc[a | b, 'label'].sum()), 'cold_total': int((a | b).sum())}
        # The legacy graph includes self-loops. A node with only a self-loop
        # is present in that graph but has no observed non-self neighbour.
        nonself = positive[positive.id1 != positive.id2]
        observed = set(nonself[['id1', 'id2']].to_numpy().ravel())
        cold = ~vp.id1.isin(observed) | ~vp.id2.isin(observed)
        report[name]['cold_excluding_self_evidence'] = int(cold.sum())
    if args.swap:
        if args.limit is not None and args.limit <= 0:
            raise ValueError('--limit must be positive')
        selected = val if args.limit is None else val[:args.limit]
        p = pairs.iloc[selected]
        model_path = Path(f'{args.checkpoints}/cascade.joblib')
        model = joblib.load(model_path)
        model.tier2.clf.n_jobs = 1
        x = np.load(f'{args.interim}/pos_train.npy', mmap_mode='r')[selected]
        s = pd.read_csv(f'{args.interim}/sentence_emb_train.csv', index_col='id')
        if not s.index.equals(pairs.index):
            raise ValueError('Sentence cache ID mismatch')
        scores = s.iloc[selected].st_score.to_numpy()
        a, ta, pa = model.predict(h.iloc[selected], x, scores, p)
        print('Original cached predictions complete', flush=True)
        b, tb, pb = model.predict(h.iloc[selected], swap_pos(x), scores,
                                   p.rename(columns={'id1': 'id2', 'id2': 'id1'}))
        saved = pd.read_csv(f'{args.predictions}/cascade_val_tiers.csv', index_col='id').loc[p.index]
        if not np.array_equal(a, saved.y_pred) or not np.array_equal(ta, saved.tier_used):
            raise ValueError('Checkpoint does not reproduce saved labels/routes')
        report['swap'] = swap_summary(a, b, pa, pb)
        report['swap'].update(checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                               saved_prediction_mismatches=0, tier_changes=int((ta != tb).sum()),
                               reversed_label_accuracy_evaluated=False)
        report['swap']['by_original_class'] = {
            str(label): swap_summary(a[p.label == label], b[p.label == label],
                                      pa[p.label == label], pb[p.label == label]) for label in (0, 1)}
        if args.pairs_output:
            path = Path(args.pairs_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'id': p.index, 'y_true_original': p.label.to_numpy(),
                          'original_prediction': a, 'reversed_prediction': b,
                          'original_score': pa, 'reversed_score': pb,
                          'original_tier': ta, 'reversed_tier': tb}).to_csv(path, index=False)
    out = Path(args.output)
    if out.exists() and not args.swap:
        previous = json.loads(out.read_text())
        if previous.get('pairs_sha256') == report['pairs_sha256'] and 'swap' in previous:
            report['swap'] = previous['swap']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
