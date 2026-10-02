"""
Generate publication figures from latex/shared/results/summary_stats.json and
save them as vector PDFs in latex/shared/figures/.

Reads:
    latex/shared/results/summary_stats.json
Writes:
    latex/shared/figures/

Usage:
    python -m scripts.thesis.generate_figures
"""

import json
import pathlib

import matplotlib
import numpy as np

from src.utils.log_utils import setup_logging

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,
})

log = setup_logging('generate_figures')

RESULTS  = pathlib.Path('latex/shared/results')
OUT_FIGS = pathlib.Path('latex/shared/figures')

PALETTE = {
    'pos': '#2196F3',
    'neg': '#F44336',
    'colors': ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#795548'],
}

DIFF_LABELS = {
    'trivial_self_loop':    'Αυτοβρόχοι',
    'trivial_high_cn':      'Υψηλό CN',
    'trivial_high_textsim': 'Υψηλή Ομοιότητα',
    'hard':                 'Δύσκολα',
}
DIFF_ORDER = ['trivial_self_loop', 'trivial_high_cn', 'trivial_high_textsim', 'hard']


def _save(fig: plt.Figure, name: str) -> None:
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    src = (OUT_FIGS / name).with_suffix('.pdf')
    fig.savefig(src, bbox_inches='tight')
    log.info('Saved → %s', src)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: separability distributions
# ---------------------------------------------------------------------------

def fig_separability(figs: dict) -> None:
    log.info('Generating separability_distributions.pdf ...')
    sep = figs.get('separability')
    if sep is None:
        log.warning('SKIP: no separability stats')
        return

    cn_edges = np.array(sep['cn_bin_edges'])
    tfidf_edges = np.array(sep['tfidf_bin_edges'])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].stairs(sep['cn_hist_pos'], cn_edges, fill=True, alpha=0.7, color=PALETTE['pos'], label='Θετικά')
    axes[0].stairs(sep['cn_hist_neg'], cn_edges, fill=True, alpha=0.7, color=PALETTE['neg'], label='Αρνητικά')
    axes[0].set_xlabel('Κοινοί Γείτονες (CN)')
    axes[0].set_ylabel('Πυκνότητα')
    axes[0].set_title('Κατανομή Κοινών Γειτόνων')
    axes[0].legend()
    axes[0].set_yscale('log')

    axes[1].stairs(sep['tfidf_hist_pos'], tfidf_edges, fill=True, alpha=0.7, color=PALETTE['pos'], label='Θετικά')
    axes[1].stairs(sep['tfidf_hist_neg'], tfidf_edges, fill=True, alpha=0.7, color=PALETTE['neg'], label='Αρνητικά')
    axes[1].set_xlabel('Ομοιότητα TF-IDF')
    axes[1].set_ylabel('Πυκνότητα')
    axes[1].set_title('Κατανομή Ομοιότητας TF-IDF')
    axes[1].legend()

    fig.tight_layout()
    _save(fig, 'separability_distributions')


# ---------------------------------------------------------------------------
# Figure 2: difficulty breakdown
# ---------------------------------------------------------------------------

def fig_difficulty(figs: dict) -> None:
    log.info('Generating difficulty_breakdown.pdf ...')
    pcts_by_cat = figs.get('difficulty_train_pct')
    if pcts_by_cat is None:
        log.warning('SKIP: no difficulty_train_pct stats')
        return

    pcts = [pcts_by_cat[cat] for cat in DIFF_ORDER]
    names = [DIFF_LABELS[cat] for cat in DIFF_ORDER]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    left = 0
    for pct, name, color in zip(pcts, names, PALETTE['colors']):
        ax.barh(0, pct, left=left, color=color, label=f'{name} ({pct:.1f}%)', height=0.5)
        if pct > 2:
            ax.text(left + pct / 2, 0, f'{pct:.1f}%',
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Ποσοστό (%)')
    ax.set_title('Κατανομή Δυσκολίας — Σύνολο Εκπαίδευσης')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    _save(fig, 'difficulty_breakdown')


# ---------------------------------------------------------------------------
# Figure 9: threshold ablation curves
# ---------------------------------------------------------------------------

def fig_threshold_ablation(figs: dict) -> None:
    log.info('Generating threshold_ablation.pdf ...')
    grid = figs.get('ablation_grid')
    if not grid:
        log.warning('SKIP: no ablation_grid stats')
        return

    tau1s = sorted({row['tau1'] for row in grid})
    # Panel A stays readable with the original (moderate) τ₂ values only — the
    # extended near-1.0 values are visualized in panel C instead.
    tau2s_legend = sorted({row['tau2'] for row in grid if row['tau2'] <= 0.9})

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4))

    for i, t2 in enumerate(tau2s_legend):
        rows = sorted((row for row in grid if row['tau2'] == t2), key=lambda r: r['tau1'])
        axes[0].plot([r['tau1'] for r in rows], [r['macro_f1'] for r in rows], marker='o',
                     color=PALETTE['colors'][i % len(PALETTE['colors'])], label=f'$\\tau_2$={t2}')
    axes[0].set_xlabel('$\\tau_1$')
    axes[0].set_ylabel('Macro F1')
    axes[0].set_title('Macro F1 vs. $\\tau_1$')
    axes[0].legend(fontsize=8)

    t1_08 = sorted((row for row in grid if row['tau1'] == 0.8), key=lambda r: r['tau2'])
    axes[1].plot([r['tau2'] for r in t1_08], [r['tier3_pct'] for r in t1_08], marker='o', color=PALETTE['neg'])
    axes[1].set_xlabel('$\\tau_2$')
    axes[1].set_ylabel('Ποσοστό κλήσεων Επιπέδου 3 (%)')
    axes[1].set_title('Ρυθμός Κλήσης Επιπέδου 3 vs. $\\tau_2$ ($\\tau_1$=0.8)')

    # Panel C — forced-escalation scenario: as τ₂→1 routes an ever-larger,
    # ever-harder-selected population to Tier 3, how does Tier 3's own Macro F1
    # (and the overall cascade Macro F1) respond? This is the "what if more
    # pairs reached Tier 3" question directly, read off the same τ₁=0.8 rows.
    t3_rate = [r['tier3_pct'] for r in t1_08]
    t3_f1   = [r['tier3_f1'] for r in t1_08]
    overall_f1 = [r['macro_f1'] for r in t1_08]
    axes[2].plot(t3_rate, t3_f1, marker='o', color=PALETTE['pos'], label='Tier-3 F1 (μόνο υποσύνολο)')
    axes[2].plot(t3_rate, overall_f1, marker='s', color=PALETTE['colors'][2], label='Συνολικό Macro F1')
    axes[2].set_xlabel('Ποσοστό κλήσεων Επιπέδου 3 (%)')
    axes[2].set_ylabel('Macro F1')
    axes[2].set_title('Επίδοση vs. Όγκος Επιπέδου 3 ($\\tau_1$=0.8)')
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    _save(fig, 'threshold_ablation')


# ---------------------------------------------------------------------------
# Figure 11: throughput comparison, CascadeLP vs. SVM
# ---------------------------------------------------------------------------

def fig_throughput_comparison(figs: dict) -> None:
    log.info('Generating throughput_comparison.pdf ...')
    thr = figs.get('throughput')
    if thr is None:
        log.warning('SKIP: no throughput stats')
        return

    names = ['CascadeLP', 'SVM (TF-IDF)']
    values = [thr['cascade_ms_per_pair'], thr['svm_ms_per_pair']]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, values, color=[PALETTE['colors'][0], PALETTE['colors'][3]], width=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('ms / ζεύγος (λογαριθμική κλίμακα)')
    ax.set_title('Λανθάνων Χρόνος Συμπερασμού ανά Ζεύγος')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f} ms', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    _save(fig, 'throughput_comparison')


# ---------------------------------------------------------------------------
# Revised-thesis figures
# ---------------------------------------------------------------------------

def fig_negative_sampling_artifact(figs: dict) -> None:
    """Visualize the endpoint-identity shortcut in DSAA 2023."""
    data = figs.get('negative_sampling_artifact')
    if not data:
        log.warning('SKIP: no negative_sampling_artifact stats')
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    groups = [f"{data['hub_count']} κόμβοι-πηγές", 'Όλοι οι υπόλοιποι']
    negative = [data['hub_negative_pct'], 100 - data['non_hub_positive_pct']]
    positive = [100 - data['hub_negative_pct'], data['non_hub_positive_pct']]
    x = np.arange(2)
    axes[0].bar(x, negative, color=PALETTE['neg'], label='Αρνητικά')
    axes[0].bar(x, positive, bottom=negative, color=PALETTE['pos'], label='Θετικά')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel('Ποσοστό ετικετών (%)')
    axes[0].set_title('Ετικέτα ανά ομάδα πρώτου άκρου')
    axes[0].legend(loc='center right')
    for i, value in enumerate([data['hub_negative_pct'], data['non_hub_positive_pct']]):
        axes[0].text(i, 50, f'{value:.3f}%', ha='center', va='center',
                     color='white', fontweight='bold')

    coverage = [data['hub_row_pct'], data['test_hub_row_pct']]
    bars = axes[1].bar(['Εκπαίδευση', 'Έλεγχος'], coverage,
                       color=[PALETTE['colors'][2], PALETTE['colors'][4]], width=0.55)
    axes[1].set_ylim(0, 65)
    axes[1].set_ylabel('Γραμμές με κόμβο-πηγή (%)')
    axes[1].set_title('Κάλυψη του ίδιου συνόλου κόμβων')
    for bar, value in zip(bars, coverage):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1,
                     f'{value:.2f}%', ha='center')
    axes[1].text(0.5, 0.04, f"Lookup Macro F1 = {data['lookup_macro_f1']:.6f}",
                 transform=axes[1].transAxes, ha='center', fontsize=9)

    fig.tight_layout()
    _save(fig, 'negative_sampling_artifact')


def fig_graph_protocol_leakage(figs: dict) -> None:
    """Contrast legacy full-graph features with a training-only graph."""
    data = figs.get('graph_protocol_leakage')
    if not data:
        log.warning('SKIP: no graph_protocol_leakage stats')
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = ['Θετικές ακμές επικύρωσης\nήδη στο πλήρες γράφημα',
             'Θετικά ζεύγη με μη παρατηρημένο άκρο\nστο γράφημα εκπαίδευσης']
    values = [data['validation_positives_in_graph'], data['training_graph_cold_positive']]
    bars = ax.bar(names, values, color=[PALETTE['neg'], PALETTE['colors'][2]], width=0.58)
    ax.set_ylabel('Πλήθος ζευγών')
    ax.set_title('Επίδραση της κατασκευής του γραφήματος στο DSAA 2023')
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis='x', labelsize=9)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025,
                f'{value:,}', ha='center', fontweight='bold')
    ax.text(0.98, 0.96, f"n επικύρωσης = {data['n_val']:,}", transform=ax.transAxes,
            ha='right', va='top', fontsize=9)
    fig.tight_layout()
    _save(fig, 'graph_protocol_leakage')


def fig_endpoint_swap(figs: dict) -> None:
    """Show prediction instability after reversing undirected endpoints."""
    data = figs.get('endpoint_swap')
    if not data:
        log.warning('SKIP: no endpoint_swap stats')
        return

    names = ['Όλα τα ζεύγη', 'Αρχικά αρνητικά', 'Αρχικά θετικά']
    values = [data['overall_change_pct'], data['negative_change_pct'], data['positive_change_pct']]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(names, values,
                  color=[PALETTE['colors'][2], PALETTE['neg'], PALETTE['pos']], width=0.58)
    ax.set_ylim(0, 108)
    ax.set_ylabel('Προβλέψεις που αλλάζουν (%)')
    ax.set_title('Ευαισθησία του CascadeLP στην αντιστροφή των άκρων')
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2,
                f'{value:.2f}%', ha='center', fontweight='bold')
    ax.text(0.98, 0.93,
            f"Μέση |Δp| = {data['mean_abs_probability_change']:.3f}",
            transform=ax.transAxes, ha='right', fontsize=9)
    fig.tight_layout()
    _save(fig, 'endpoint_swap_sensitivity')


def fig_cross_dataset_performance(figs: dict) -> None:
    """Compare model rankings under the DSAA and Wiki-CS-8k protocols."""
    data = figs.get('cross_dataset_performance')
    if not data:
        log.warning('SKIP: no cross_dataset_performance stats')
        return

    labels = ['Δομικό', 'TF-IDF', 'POS', 'Embedding', 'SVM', 'CascadeLP']
    colors = [PALETTE['colors'][1], PALETTE['colors'][4], PALETTE['colors'][2],
              PALETTE['colors'][5], PALETTE['colors'][3], PALETTE['colors'][0]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    datasets = [('DSAA 2023 — αρχικό πρωτόκολλο', data['dsaa_macro_f1']),
                ('Wiki-CS-8k — αυστηρό πρωτόκολλο', data['wiki_macro_f1'])]
    x = np.arange(len(labels))
    for ax, (title, values) in zip(axes, datasets):
        bars = ax.bar(x, values, color=colors, width=0.68)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right')
        ax.set_ylim(0.75, 1.015)
        ax.set_title(title)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004,
                    f'{value:.3f}', ha='center', fontsize=8, rotation=90)

    ci = data.get('wiki_cascade_ci95')
    if ci:
        center = data['wiki_macro_f1'][-1]
        axes[1].errorbar([x[-1]], [center],
                         yerr=[[center - ci[0]], [ci[1] - center]],
                         fmt='none', ecolor='black', capsize=4, linewidth=1.2)
    axes[0].set_ylabel('Macro F1')
    fig.tight_layout()
    _save(fig, 'cross_dataset_performance')


def fig_routing_comparison(figs: dict) -> None:
    """Compare CascadeLP final-decision routing across the two datasets."""
    data = figs.get('routing_comparison')
    if not data:
        log.warning('SKIP: no routing_comparison stats')
        return

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    labels = ['DSAA 2023', 'Wiki-CS-8k']
    left = np.zeros(2)
    tier_labels = ['Επίπεδο 0', 'Επίπεδο 1 — Δομή', 'Επίπεδο 2 — POS',
                   'Επίπεδο 3 — Embedding']
    for tier, tier_label in enumerate(tier_labels):
        values = np.array([data['dsaa'][tier], data['wiki'][tier]])
        ax.barh(labels, values, left=left, color=PALETTE['colors'][tier], label=tier_label)
        for row, (start, value) in enumerate(zip(left, values)):
            if value >= 4:
                ax.text(start + value / 2, row, f'{value:.1f}%', ha='center', va='center',
                        color='white', fontweight='bold', fontsize=9)
        left += values
    ax.set_xlim(0, 100)
    ax.set_xlabel('Ζεύγη ανά επίπεδο τελικής απόφασης (%)')
    ax.set_title('Μεταβολή της δρομολόγησης μεταξύ συνόλων δεδομένων')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)
    fig.tight_layout()
    _save(fig, 'routing_comparison')


def fig_wiki_diagnostic_subsets(figs: dict) -> None:
    """Show both size and performance of the Wiki-CS-8k diagnostic slices."""
    data = figs.get('wiki_diagnostic_subsets')
    if not data:
        log.warning('SKIP: no wiki_diagnostic_subsets stats')
        return

    keys = ['zero_cn', 'functional_cold_start', 'missing_text']
    labels = ['Μηδενικό CN', 'Cold-start', 'Ελλιπές κείμενο']
    macro_f1 = [data[key]['macro_f1'] for key in keys]
    recall = [data[key]['positive_recall'] for key in keys]
    counts = [data[key]['n'] for key in keys]
    x = np.arange(len(keys))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars_f1 = ax.bar(x - width / 2, macro_f1, width, label='Macro F1', color=PALETTE['pos'])
    bars_rec = ax.bar(x + width / 2, recall, width, label='Ανάκληση θετικών',
                      color=PALETTE['colors'][2])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{label}\n(n={count:,})' for label, count in zip(labels, counts)])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Τιμή μετρικής')
    ax.set_title('Διαγνωστικά υποσύνολα στο Wiki-CS-8k')
    ax.legend()
    for bars in (bars_f1, bars_rec):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{bar.get_height():.3f}', ha='center', fontsize=8)
    fig.tight_layout()
    _save(fig, 'wiki_diagnostic_subsets')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_SPARSE_CONDITION_LABELS = {
    ('random', 'random'): 'Εκπ. τυχαία\nΈλεγχος τυχαία',
    ('random', 'hard'):   'Εκπ. τυχαία\nΈλεγχος δύσκολα',
    ('mixed', 'random'):  'Εκπ. μικτά\nΈλεγχος τυχαία',
    ('mixed', 'hard'):    'Εκπ. μικτά\nΈλεγχος δύσκολα',
}


def fig_sparse_model_comparison(figs: dict) -> None:
    """Every model's Macro F1 in the four sparse train/test negative-sampling conditions."""
    data = figs.get('sparse_stress')
    if not data:
        log.warning('SKIP: no sparse_stress stats')
        return

    labels = ['Δομικό', 'TF-IDF', 'POS', 'Embedding', 'SVM', 'CascadeLP']
    colors = [PALETTE['colors'][1], PALETTE['colors'][4], PALETTE['colors'][2],
              PALETTE['colors'][5], PALETTE['colors'][3], PALETTE['colors'][0]]
    conditions = data['conditions']
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    x = np.arange(len(conditions))
    width = 0.13
    for i, (label, color) in enumerate(zip(labels, colors)):
        values = [c['macro_f1'][i] for c in conditions]
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, color=color, label=label,
                      edgecolor='black' if label == 'CascadeLP' else 'none', linewidth=0.8)
        if label == 'CascadeLP':
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f'{value:.3f}',
                        ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_SPARSE_CONDITION_LABELS[(c['train'], c['test'])] for c in conditions])
    ax.set_ylim(0.2, 0.95)
    ax.set_ylabel('Macro F1')
    ax.set_title('Αραιό γράφημα: όλα τα μοντέλα ανά συνθήκη αρνητικών')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=6)
    fig.tight_layout()
    _save(fig, 'sparse_model_comparison')


def fig_sparse_routing(figs: dict) -> None:
    """CascadeLP final-decision routing: dense Wiki graph vs. the four sparse conditions."""
    data = figs.get('sparse_stress')
    if not data:
        log.warning('SKIP: no sparse_stress stats')
        return

    rows = [(_SPARSE_CONDITION_LABELS[(c['train'], c['test'])].replace('\n', ' · '),
             c['tier_pct']) for c in data['conditions']]
    if data.get('dense_tier_pct'):
        rows.insert(0, ('Πυκνό γράφημα', data['dense_tier_pct']))
    rows.reverse()
    labels = [r[0] for r in rows]
    tier_labels = ['Επίπεδο 0', 'Επίπεδο 1 — Δομή', 'Επίπεδο 2 — POS',
                   'Επίπεδο 3 — Embedding']
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    left = np.zeros(len(rows))
    for tier, tier_label in enumerate(tier_labels):
        values = np.array([r[1][tier] for r in rows])
        ax.barh(labels, values, left=left, color=PALETTE['colors'][tier], label=tier_label)
        for row, (start, value) in enumerate(zip(left, values)):
            if value >= 5:
                ax.text(start + value / 2, row, f'{value:.1f}%', ha='center', va='center',
                        color='white', fontweight='bold', fontsize=9)
        left += values
    ax.set_xlim(0, 100)
    ax.set_xlabel('Ζεύγη ανά επίπεδο τελικής απόφασης (%)')
    ax.set_title('Δρομολόγηση του CascadeLP: πυκνό έναντι αραιού γραφήματος')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=4)
    fig.tight_layout()
    _save(fig, 'sparse_routing')


def main() -> None:
    stats_path = RESULTS / 'summary_stats.json'
    if not stats_path.exists():
        raise SystemExit(
            f'{stats_path} not found — run `make compute-stats` first '
            '(requires the full local data/feature/train pipeline output).'
        )
    stats = json.loads(stats_path.read_text())
    figs = stats['figures']

    fig_separability(figs)
    fig_difficulty(figs)
    fig_threshold_ablation(figs)
    fig_throughput_comparison(figs)
    fig_negative_sampling_artifact(figs)
    fig_graph_protocol_leakage(figs)
    fig_endpoint_swap(figs)
    fig_cross_dataset_performance(figs)
    fig_routing_comparison(figs)
    fig_wiki_diagnostic_subsets(figs)
    fig_sparse_model_comparison(figs)
    fig_sparse_routing(figs)
    log.info('Done.')


if __name__ == '__main__':
    main()
