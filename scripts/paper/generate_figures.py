"""
Generate figures from outputs/stats/summary_stats.json and save to outputs/figures/.

Reads:
    outputs/stats/summary_stats.json
Writes:
    outputs/figures/

Usage:
    python -m scripts.paper.generate_figures
"""

import json
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.utils.log_utils import setup_logging

matplotlib.use('Agg')

log = setup_logging('generate_figures')

STATS    = pathlib.Path('outputs/stats')
OUT_FIGS = pathlib.Path('outputs/figures')

PALETTE = {
    'pos': '#2196F3',
    'neg': '#F44336',
    'colors': ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#795548'],
}

TIER_NAMES = {0: 'Επίπεδο 0\n(αυτοβρόχος)', 1: 'Επίπεδο 1\n(Structural)',
              2: 'Επίπεδο 2\n(POS+RF)', 3: 'Επίπεδο 3\n(Embedding)'}

DIFF_LABELS = {
    'trivial_self_loop':    'Αυτοβρόχοι',
    'trivial_high_cn':      'Υψηλό CN',
    'trivial_high_textsim': 'Υψηλή Ομοιότητα',
    'hard':                 'Δύσκολα',
}
DIFF_ORDER = ['trivial_self_loop', 'trivial_high_cn', 'trivial_high_textsim', 'hard']


def _save(fig: plt.Figure, name: str) -> None:
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    src = OUT_FIGS / name
    fig.savefig(src, dpi=150, bbox_inches='tight')
    log.info('Saved → %s', src)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: separability distributions
# ---------------------------------------------------------------------------

def fig_separability(figs: dict) -> None:
    log.info('Generating separability_distributions.png ...')
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
    _save(fig, 'separability_distributions.png')


# ---------------------------------------------------------------------------
# Figure 2: difficulty breakdown
# ---------------------------------------------------------------------------

def fig_difficulty(figs: dict) -> None:
    log.info('Generating difficulty_breakdown.png ...')
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
    _save(fig, 'difficulty_breakdown.png')


# ---------------------------------------------------------------------------
# Figure 3: SVM metrics
# ---------------------------------------------------------------------------

def fig_svm_metrics(figs: dict) -> None:
    log.info('Generating svm_metrics.png ...')
    svm = figs.get('svm_metrics')
    if svm is None:
        log.warning('SKIP: no svm_metrics')
        return

    lat = svm.get('latency_ms', 0)
    n   = svm.get('n_val', 1)
    lat_per_pair = lat / n if n else 0

    metrics = [
        ('Macro F1',        svm.get('macro_f1', 0)),
        ('AUC-ROC',         svm.get('auc_roc', 0)),
        ('Cold-start F1',   svm.get('cold_start_f1', 0)),
        ('Latency (ms/pair)', lat_per_pair),
    ]
    names  = [m[0] for m in metrics]
    values = [m[1] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=PALETTE['colors'], width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, max(values) * 1.15)
    ax.set_ylabel('Τιμή')
    ax.set_title('Απόδοση μοντέλου αναφοράς TF-IDF + SVM')
    fig.tight_layout()
    _save(fig, 'svm_metrics.png')


# ---------------------------------------------------------------------------
# Figure 4: cascade tier routing
# ---------------------------------------------------------------------------

def fig_tier_routing(figs: dict) -> None:
    log.info('Generating tier_routing.png ...')
    tier_counts = figs.get('tier_counts')
    n_total = figs.get('tier_counts_total')
    if not tier_counts or not n_total:
        log.warning('SKIP: no tier_counts stats')
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    left = 0
    for tier in sorted(TIER_NAMES):
        n = int(tier_counts.get(str(tier), 0))
        pct = 100 * n / n_total
        color = PALETTE['colors'][tier % len(PALETTE['colors'])]
        ax.barh(0, pct, left=left, color=color,
                label=f'{TIER_NAMES[tier].splitlines()[0]} ({pct:.1f}%, n={n:,})', height=0.5)
        if pct > 2:
            ax.text(left + pct / 2, 0, f'{pct:.1f}%',
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Ποσοστό ζευγών (%)')
    ax.set_title('Δρομολόγηση Ζευγών ανά Επίπεδο CascadeLP')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, 'tier_routing.png')


# ---------------------------------------------------------------------------
# Figure 5: tier x difficulty accuracy heatmap
# ---------------------------------------------------------------------------

def fig_tier_difficulty_heatmap(figs: dict) -> None:
    log.info('Generating tier_difficulty_heatmap.png ...')
    table = figs.get('tier_difficulty_table')
    if not table:
        log.warning('SKIP: no tier_difficulty_table stats')
        return

    tiers = figs['tiers_present']
    diffs = figs['diffs_present']

    acc = np.full((len(tiers), len(diffs)), np.nan)
    for i, tier in enumerate(tiers):
        for j, diff in enumerate(diffs):
            cell = table.get(str(tier), {}).get(diff)
            if cell:
                acc[i, j] = cell['acc']

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(acc, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels([DIFF_LABELS[d] for d in diffs], rotation=20, ha='right')
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels([TIER_NAMES.get(t, str(t)).replace('\n', ' ') for t in tiers])
    for i in range(len(tiers)):
        for j in range(len(diffs)):
            if not np.isnan(acc[i, j]):
                ax.text(j, i, f'{acc[i, j]:.3f}', ha='center', va='center', fontsize=9)
    ax.set_title('Ακρίβεια ανά Επίπεδο × Κατηγορία Δυσκολίας')
    fig.colorbar(im, ax=ax, label='Ακρίβεια')
    fig.tight_layout()
    _save(fig, 'tier_difficulty_heatmap.png')


# ---------------------------------------------------------------------------
# Figure 6: per-tier confusion matrices
# ---------------------------------------------------------------------------

def fig_confusion_matrices(figs: dict) -> None:
    log.info('Generating tier_confusion_matrices.png ...')
    cms = figs.get('confusion_matrices')
    if not cms:
        log.warning('SKIP: no confusion_matrices stats')
        return

    tiers = figs['tiers_present']
    fig, axes = plt.subplots(1, len(tiers), figsize=(3.2 * len(tiers), 3.2))
    if len(tiers) == 1:
        axes = [axes]

    for ax, tier in zip(axes, tiers):
        cm = np.array(cms[str(tier)])
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['0', '1'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['0', '1'])
        ax.set_xlabel('Πρόβλεψη')
        ax.set_ylabel('Πραγματική')
        ax.set_title(TIER_NAMES.get(tier, str(tier)).replace('\n', ' '), fontsize=9)

    fig.suptitle('Πίνακες Σύγχυσης ανά Επίπεδο')
    fig.tight_layout()
    _save(fig, 'tier_confusion_matrices.png')


# ---------------------------------------------------------------------------
# Figure 7: ROC curve, CascadeLP vs. SVM
# ---------------------------------------------------------------------------

def fig_roc_curve(figs: dict) -> None:
    log.info('Generating roc_curve.png ...')
    roc_cascade = figs.get('roc_cascade')
    roc_svm = figs.get('roc_svm')

    fig, ax = plt.subplots(figsize=(6, 5.5))
    plotted = False

    if roc_cascade is not None:
        ax.plot(roc_cascade['fpr'], roc_cascade['tpr'], color=PALETTE['pos'], label='CascadeLP')
        plotted = True
    if roc_svm is not None:
        ax.plot(roc_svm['fpr'], roc_svm['tpr'], color=PALETTE['neg'], label='SVM (TF-IDF)')
        plotted = True

    if not plotted:
        log.warning('SKIP: no scored predictions available for ROC curve')
        plt.close(fig)
        return

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Καμπύλη ROC')
    ax.legend(loc='lower right')
    fig.tight_layout()
    _save(fig, 'roc_curve.png')


# ---------------------------------------------------------------------------
# Figure 8: per-tier confidence-score distribution
# ---------------------------------------------------------------------------

def fig_confidence_distribution(figs: dict) -> None:
    log.info('Generating tier_confidence_distribution.png ...')
    hist = figs.get('confidence_hist')
    if not hist:
        log.warning('SKIP: no confidence_hist stats')
        return

    tiers = sorted(int(t) for t in hist)
    fig, axes = plt.subplots(1, len(tiers), figsize=(3.5 * len(tiers), 3.2), sharey=True)
    if len(tiers) == 1:
        axes = [axes]

    for ax, tier in zip(axes, tiers):
        h = hist[str(tier)]
        edges = np.array(h['bin_edges'])
        widths = np.diff(edges)
        ax.bar(edges[:-1], h['counts'], width=widths, align='edge',
               color=PALETTE['colors'][tier % len(PALETTE['colors'])])
        ax.set_title(TIER_NAMES.get(tier, str(tier)).replace('\n', ' '), fontsize=9)
        ax.set_xlabel('P(y=1)')
    axes[0].set_ylabel('Πλήθος ζευγών')

    fig.suptitle('Κατανομή Βαθμολογίας Εμπιστοσύνης ανά Επίπεδο')
    fig.tight_layout()
    _save(fig, 'tier_confidence_distribution.png')


# ---------------------------------------------------------------------------
# Figure 9: threshold ablation curves
# ---------------------------------------------------------------------------

def fig_threshold_ablation(figs: dict) -> None:
    log.info('Generating threshold_ablation.png ...')
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
    _save(fig, 'threshold_ablation.png')


# ---------------------------------------------------------------------------
# Figure 11: throughput comparison, CascadeLP vs. SVM
# ---------------------------------------------------------------------------

def fig_throughput_comparison(figs: dict) -> None:
    log.info('Generating throughput_comparison.png ...')
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
    _save(fig, 'throughput_comparison.png')


# ---------------------------------------------------------------------------
# Figure 12: cold-start comparison, CascadeLP vs. SVM
# ---------------------------------------------------------------------------

def fig_coldstart_comparison(figs: dict) -> None:
    log.info('Generating coldstart_comparison.png ...')
    cs = figs.get('coldstart')
    if cs is None:
        log.warning('SKIP: no coldstart stats')
        return

    names = ['CascadeLP', 'SVM (TF-IDF)']
    overall = [cs['cascade']['macro_f1'], cs['svm']['macro_f1']]
    cold_start = [cs['cascade']['cold_start_f1'], cs['svm']['cold_start_f1']]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, overall, width, label='Macro F1 (σύνολο)', color=PALETTE['colors'][0])
    ax.bar(x + width / 2, cold_start, width, label='Macro F1 (cold-start)', color=PALETTE['colors'][2])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title('Απόδοση σε Ζεύγη Cold-Start')
    ax.legend()
    fig.tight_layout()
    _save(fig, 'coldstart_comparison.png')


# ---------------------------------------------------------------------------
# Figure 13: error breakdown by difficulty, CascadeLP vs. SVM
# ---------------------------------------------------------------------------

def fig_error_by_difficulty_comparison(figs: dict) -> None:
    log.info('Generating error_by_difficulty_comparison.png ...')
    err = figs.get('error_by_difficulty')
    if not err or 'svm' not in err:
        log.warning('SKIP: need both cascade and svm error_by_difficulty stats')
        return

    diffs = figs['diffs_present']
    cascade_err_pct = [err['cascade'][d] for d in diffs]
    svm_err_pct = [err['svm'][d] for d in diffs]

    x = np.arange(len(diffs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, cascade_err_pct, width, label='CascadeLP', color=PALETTE['colors'][0])
    ax.bar(x + width / 2, svm_err_pct, width, label='SVM (TF-IDF)', color=PALETTE['colors'][3])
    ax.set_xticks(x)
    ax.set_xticklabels([DIFF_LABELS[d] for d in diffs], rotation=15, ha='right')
    ax.set_ylabel('Ποσοστό Σφάλματος (%)')
    ax.set_title('Ποσοστό Σφάλματος ανά Κατηγορία Δυσκολίας')
    ax.legend()
    fig.tight_layout()
    _save(fig, 'error_by_difficulty_comparison.png')


# ---------------------------------------------------------------------------
# Figure 14: graph degree distribution
# ---------------------------------------------------------------------------

def fig_graph_degree_distribution(figs: dict) -> None:
    log.info('Generating graph_degree_distribution.png ...')
    hist = figs.get('graph_degree_hist')
    if hist is None:
        log.warning('SKIP: no graph_degree_hist stats')
        return

    edges = np.array(hist['bin_edges'])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stairs(hist['counts'], edges, fill=True, color=PALETTE['colors'][0])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Βαθμός Κόμβου (λογαριθμική κλίμακα)')
    ax.set_ylabel('Πλήθος Κόμβων (λογαριθμική κλίμακα)')
    ax.set_title('Κατανομή Βαθμού στο Γράφημα Θετικών Ακμών')
    fig.tight_layout()
    _save(fig, 'graph_degree_distribution.png')


# ---------------------------------------------------------------------------
# Figure 15: dataset composition detail
# ---------------------------------------------------------------------------

def fig_dataset_composition(figs: dict) -> None:
    log.info('Generating dataset_composition.png ...')
    comp = figs.get('dataset_composition')
    if comp is None:
        log.warning('SKIP: no dataset_composition stats')
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    label_counts = comp['label_counts']
    neg, pos = label_counts.get('0', 0), label_counts.get('1', 0)
    axes[0].bar(['Αρνητικά', 'Θετικά'], [neg, pos],
                color=[PALETTE['neg'], PALETTE['pos']], width=0.5)
    axes[0].set_ylabel('Πλήθος Ζευγών')
    axes[0].set_title('Ισορροπία Κλάσεων — train.csv')
    for i, v in enumerate([neg, pos]):
        axes[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)

    diff_counts = comp['difficulty_counts']
    axes[1].bar([DIFF_LABELS[c] for c in DIFF_ORDER], [diff_counts.get(c, 0) for c in DIFF_ORDER],
                color=PALETTE['colors'][:len(DIFF_ORDER)], width=0.5)
    axes[1].set_ylabel('Πλήθος Ζευγών')
    axes[1].set_title('Κατηγορίες Δυσκολίας — train.csv')
    axes[1].tick_params(axis='x', rotation=15)

    fig.tight_layout()
    _save(fig, 'dataset_composition.png')


# ---------------------------------------------------------------------------
# Figure 16: DSAA 2023 leaderboard comparison
# ---------------------------------------------------------------------------

# Reported scores from other DSAA 2023 competition entries, matching
# latex/thesis/body_matter/chap2.tex's tab:dsaa-submissions exactly (external, citation-backed
# facts — not this project's experiment output, hence literal here per CLAUDE.md's
# no-magic-numbers carve-out). Teams with no reported numeric score in their short paper
# (nguyen2023mat, mata2023link, kansal2023predict) are omitted from the chart below.
_DSAA_OTHER_TEAMS = [
    ('UIT-NLP\n(phan2023link)', 1.0),
    ('Tran et al.\n(tran2023text)', 0.99999),
    ('Yang\n(yang2023achieving)', 0.99),
    ('Giannoulidis &\nMavroudopoulos', 0.948),
]


def fig_dsaa_leaderboard_comparison(figs: dict) -> None:
    log.info('Generating dsaa_leaderboard_comparison.png ...')
    our_score = figs.get('kaggle_our_score')
    if our_score is None:
        log.warning('SKIP: no kaggle_our_score stat')
        return

    names = [t for t, _ in _DSAA_OTHER_TEAMS] + ['CascadeLP\n(παρούσα εργασία)']
    scores = [s for _, s in _DSAA_OTHER_TEAMS] + [our_score]
    colors = [PALETTE['colors'][1]] * len(_DSAA_OTHER_TEAMS) + [PALETTE['colors'][0]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, scores, color=colors, width=0.55)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.5f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0.9, 1.005)
    ax.set_ylabel('Macro F1')
    ax.set_title('Σύγκριση με Άλλες Συμμετοχές DSAA 2023')
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    _save(fig, 'dsaa_leaderboard_comparison.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    stats_path = STATS / 'summary_stats.json'
    if not stats_path.exists():
        raise SystemExit(
            f'{stats_path} not found — run `make compute-stats` first '
            '(requires the full local data/feature/train pipeline output).'
        )
    stats = json.loads(stats_path.read_text())
    figs = stats['figures']

    fig_separability(figs)
    fig_difficulty(figs)
    fig_svm_metrics(figs)
    fig_tier_routing(figs)
    fig_tier_difficulty_heatmap(figs)
    fig_confusion_matrices(figs)
    fig_roc_curve(figs)
    fig_confidence_distribution(figs)
    fig_threshold_ablation(figs)
    fig_throughput_comparison(figs)
    fig_coldstart_comparison(figs)
    fig_error_by_difficulty_comparison(figs)
    fig_graph_degree_distribution(figs)
    fig_dataset_composition(figs)
    fig_dsaa_leaderboard_comparison(figs)
    log.info('Done.')


if __name__ == '__main__':
    main()
