"""Numeric aggregates consumed by the thesis figure renderer.

This is an internal helper; run ``compute_summary_stats`` as the entry point.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

_MODEL_NAMES = ('structural', 'tfidf', 'pos', 'embedding', 'svm', 'cascade')
DIFF_ORDER = ['trivial_self_loop', 'trivial_high_cn', 'trivial_high_textsim', 'hard']


def _compute_figures(d: dict, shared: dict) -> dict:
    figs: dict = {}

    # ---- separability histograms ----
    train = d['train']
    self_loop_mask = train['id1'] == train['id2']
    mask = (~self_loop_mask).values
    labels = train['label'].values[mask]
    cn = d['struct_train']['cn'].values[mask]
    tfidf_s = d['tfidf_train']['tfidf_score'].values[mask]

    rng = np.random.default_rng(d['cfg']['seed'])
    idx = rng.choice(len(labels), min(100_000, len(labels)), replace=False)
    labels, cn, tfidf_s = labels[idx], cn[idx], tfidf_s[idx]
    pos_mask, neg_mask = labels == 1, labels == 0

    cn_bins = np.arange(0, min(cn.max() + 2, 20))
    cn_pos_counts, _ = np.histogram(cn[pos_mask], bins=cn_bins, density=True)
    cn_neg_counts, _ = np.histogram(cn[neg_mask], bins=cn_bins, density=True)

    tfidf_bins = np.linspace(float(tfidf_s.min()), float(tfidf_s.max()), 51)
    tfidf_pos_counts, _ = np.histogram(tfidf_s[pos_mask], bins=tfidf_bins, density=True)
    tfidf_neg_counts, _ = np.histogram(tfidf_s[neg_mask], bins=tfidf_bins, density=True)

    figs['separability'] = {
        'cn_bin_edges': cn_bins.tolist(),
        'cn_hist_pos': cn_pos_counts.tolist(),
        'cn_hist_neg': cn_neg_counts.tolist(),
        'tfidf_bin_edges': tfidf_bins.tolist(),
        'tfidf_hist_pos': tfidf_pos_counts.tolist(),
        'tfidf_hist_neg': tfidf_neg_counts.tolist(),
    }

    # ---- difficulty breakdown (train) — percentages per DIFF_ORDER category ----
    diff_train_counts = shared['diff_train_counts']
    n_dt = len(d['diff_train'])
    figs['difficulty_train_pct'] = {
        cat: 100 * diff_train_counts.get(cat, 0) / n_dt for cat in DIFF_ORDER
    }

    # ---- dataset composition ----
    figs['dataset_composition'] = {
        'label_counts': {'0': shared['n_train_neg'], '1': shared['n_train_pos']},
        'difficulty_counts': {cat: int(diff_train_counts.get(cat, 0)) for cat in DIFF_ORDER},
    }

    # ---- svm metrics (already a small metrics dict — pass through raw) ----
    figs['svm_metrics'] = d['svm']

    # ---- cascade val tiers derived aggregates ----
    vt = d['val_tiers']
    n_vt = len(vt)
    tier_counts = vt['tier_used'].value_counts()
    figs['tier_counts'] = {str(t): int(tier_counts.get(t, 0)) for t in range(4)}
    figs['tier_counts_total'] = n_vt

    tiers_present = sorted(vt['tier_used'].unique().tolist())
    diffs_present = [c for c in DIFF_ORDER if c in vt['difficulty'].unique()]

    tier_difficulty_table = {}
    for tier in tiers_present:
        tier_difficulty_table[str(tier)] = {}
        for diff in diffs_present:
            sub = vt[(vt['tier_used'] == tier) & (vt['difficulty'] == diff)]
            if len(sub):
                tier_difficulty_table[str(tier)][diff] = {
                    'n': int(len(sub)), 'acc': float(sub['correct'].mean())
                }
    figs['tier_difficulty_table'] = tier_difficulty_table
    figs['tiers_present'] = tiers_present
    figs['diffs_present'] = diffs_present

    confusion_matrices = {}
    for tier in tiers_present:
        sub = vt[vt['tier_used'] == tier]
        cm = confusion_matrix(sub['y_true'], sub['y_pred'], labels=[0, 1])
        confusion_matrices[str(tier)] = cm.tolist()
    figs['confusion_matrices'] = confusion_matrices

    if 'score' in vt.columns:
        confidence_hist = {}
        for tier in [t for t in tiers_present if t > 0]:
            sub = vt[vt['tier_used'] == tier]
            counts, edges = np.histogram(sub['score'], bins=30)
            confidence_hist[str(tier)] = {'bin_edges': edges.tolist(), 'counts': counts.tolist()}
        figs['confidence_hist'] = confidence_hist

        fpr_grid = np.linspace(0, 1, 200)
        fpr, tpr, _ = roc_curve(vt['y_true'], vt['score'])
        figs['roc_cascade'] = {'fpr': fpr_grid.tolist(), 'tpr': np.interp(fpr_grid, fpr, tpr).tolist()}

    se = d['svm_errors']
    if se is not None and 'score' in se.columns:
        fpr_grid = np.linspace(0, 1, 200)
        fpr, tpr, _ = roc_curve(se['y_true'], se['score'])
        figs['roc_svm'] = {'fpr': fpr_grid.tolist(), 'tpr': np.interp(fpr_grid, fpr, tpr).tolist()}

    # ---- error by difficulty comparison (cascade vs. svm) ----
    error_cascade = {d_: 100 * (1 - vt[vt['difficulty'] == d_]['correct'].mean()) for d_ in diffs_present}
    figs['error_by_difficulty'] = {'cascade': error_cascade}
    if se is not None:
        error_svm = {
            d_: 100 * (1 - se[se['difficulty'] == d_]['correct'].mean()) if d_ in se['difficulty'].unique() else 0.0
            for d_ in diffs_present
        }
        figs['error_by_difficulty']['svm'] = error_svm

    # ---- graph degree distribution ----
    G = shared['G']
    degrees = np.array([deg for _, deg in G.degree()])
    bins = np.logspace(0, np.log10(max(degrees.max(), 2)), 40)
    counts, edges = np.histogram(degrees, bins=bins)
    figs['graph_degree_hist'] = {'bin_edges': edges.tolist(), 'counts': counts.tolist()}

    # ---- threshold ablation grid (already a small aggregate CSV) ----
    figs['ablation_grid'] = d['ablation'].to_dict(orient='records')

    # ---- throughput comparison ----
    thr = d['throughput']
    svm = d['svm']
    if thr and svm and svm.get('latency_ms') and svm.get('n_val'):
        figs['throughput'] = {
            'cascade_ms_per_pair': thr['ms_per_pair'],
            'svm_ms_per_pair': svm['latency_ms'] / svm['n_val'],
        }

    # ---- cold-start comparison ----
    cascade_metrics = d['cascade_val_metrics']
    if cascade_metrics and svm:
        figs['coldstart'] = {
            'cascade': {'macro_f1': cascade_metrics['macro_f1'], 'cold_start_f1': cascade_metrics['cold_start_f1']},
            'svm': {'macro_f1': svm['macro_f1'], 'cold_start_f1': svm['cold_start_f1']},
        }

    # ---- DSAA leaderboard comparison ----
    if shared['kaggle_our_private'] is not None:
        figs['kaggle_our_score'] = shared['kaggle_our_private']

    # ---- Publication figures for the revised thesis argument ----
    neg_audit = d['neg_sampling_audit']
    if neg_audit:
        reference = next(
            row for row in neg_audit['by_threshold']
            if row['threshold'] == neg_audit['reference_threshold']
        )
        figs['negative_sampling_artifact'] = {
            'hub_count': reference['n_hub_id1'],
            'unique_source_count': reference['n_unique_id1'],
            'hub_row_pct': reference['hub_row_pct'],
            'test_hub_row_pct': neg_audit['test_rows_with_hub_id1_pct'],
            'hub_negative_pct': 100 * reference['hub_negative_rate'],
            'non_hub_positive_pct': 100 * reference['non_hub_positive_rate'],
            'lookup_macro_f1': reference['trivial_lookup_macro_f1'],
        }

    protocol = d['protocol_audit']
    if protocol:
        figs['graph_protocol_leakage'] = {
            'n_val': protocol['n_val'],
            'validation_positives_in_graph': protocol['validation_positives_in_graph'],
            'full_graph_cold_positive': protocol['full_graph']['cold_positive'],
            'training_graph_cold_positive': protocol['training_partition_graph']['cold_positive'],
        }
        swap = protocol.get('swap')
        if swap:
            figs['endpoint_swap'] = {
                'overall_change_pct': swap['label_change_pct'],
                'negative_change_pct': swap['by_original_class']['0']['label_change_pct'],
                'positive_change_pct': swap['by_original_class']['1']['label_change_pct'],
                'mean_abs_probability_change': swap['mean_abs_probability_change'],
            }

    figs.update(_sparse_figures(d))

    wikipedia = d['wikipedia']
    dsaa_metrics = d['baseline_metrics']
    if wikipedia and all(dsaa_metrics.get(name) for name in _MODEL_NAMES):
        figs['cross_dataset_performance'] = {
            'models': list(_MODEL_NAMES),
            'dsaa_macro_f1': [dsaa_metrics[name]['macro_f1'] for name in _MODEL_NAMES],
            'wiki_macro_f1': [wikipedia[name]['macro_f1'] for name in _MODEL_NAMES],
            'wiki_cascade_ci95': wikipedia.get('bootstrap', {}).get('cascade_macro_f1_ci95'),
        }

        wiki_tiers = wikipedia['cascade']['tier_stats']
        figs['routing_comparison'] = {
            'dsaa': [100 * int(tier_counts.get(tier, 0)) / n_vt for tier in range(4)],
            'wiki': [wiki_tiers[f'tier{tier}']['pct'] for tier in range(4)],
        }

        diagnostics = wikipedia['cascade']['diagnostic_subsets']
        figs['wiki_diagnostic_subsets'] = {
            key: {
                'n': diagnostics[key]['n'],
                'macro_f1': diagnostics[key]['macro_f1'],
                'positive_recall': diagnostics[key]['positive_recall'],
            }
            for key in ('zero_cn', 'functional_cold_start', 'missing_text')
        }

    return figs


def _sparse_figures(d: dict) -> dict:
    """Per-condition model scores and CascadeLP routing for the sparse Wiki stress test."""
    sparse_random, sparse_mixed = d.get('wiki_sparse_random'), d.get('wiki_sparse_mixed')
    if not (sparse_random and sparse_mixed):
        return {}
    conditions = []
    for run, train in ((sparse_random, 'random'), (sparse_mixed, 'mixed')):
        for suite, test in (('test_random', 'random'), ('test_hard', 'hard')):
            results = run['suites'][suite]
            tiers = results['cascade']['tier_stats']
            conditions.append({
                'train': train,
                'test': test,
                'macro_f1': [results[name]['macro_f1'] for name in _MODEL_NAMES],
                'tier_pct': [tiers[f'tier{tier}']['pct'] for tier in range(4)],
                'tier1_errors': results['cascade']['tier_performance']['tier1']['errors'],
            })
    figs = {'sparse_stress': {'models': list(_MODEL_NAMES), 'conditions': conditions}}
    wikipedia = d.get('wikipedia')
    if wikipedia:
        wiki_tiers = wikipedia['cascade']['tier_stats']
        figs['sparse_stress']['dense_tier_pct'] = [
            wiki_tiers[f'tier{tier}']['pct'] for tier in range(4)]
    return figs
