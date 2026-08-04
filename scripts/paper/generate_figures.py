"""
Generate thesis figures from experiment data and save to outputs/figures/ and paper/figures/.

Every figure reads only persisted data/interim/ or outputs/predictions/ artifacts —
none of them call a model live — so figure regeneration never requires re-running the
training/evaluation pipeline.

Produces (existing):
  separability_distributions.png  — CN and TF-IDF distributions for pos vs. neg pairs
  difficulty_breakdown.png        — difficulty-category bar chart (train set)
  svm_metrics.png                 — SVM baseline performance bar chart

Produces (new):
  tier_routing.png                 — CascadeLP tier routing breakdown (n / % per tier)
  tier_difficulty_heatmap.png      — accuracy heatmap, tier x difficulty
  tier_confusion_matrices.png      — per-tier confusion matrices
  roc_curve.png                    — ROC curve, CascadeLP vs. SVM baseline
  tier_confidence_distribution.png — per-tier confidence-score distribution
  threshold_ablation.png           — Macro F1 vs. tau1/tau2, Tier-3 call rate vs. tau2
  node2vec_ablation.png            — with/without Node2Vec comparison
  throughput_comparison.png        — CascadeLP vs. SVM inference latency
  coldstart_comparison.png         — cold-start Macro F1, CascadeLP vs. SVM
  error_by_difficulty_comparison.png — error rate by difficulty, CascadeLP vs. SVM
  graph_degree_distribution.png    — positive-edge graph degree histogram
  dataset_composition.png          — train class balance + difficulty composition
  dsaa_leaderboard_comparison.png  — Macro F1 vs. other DSAA 2023 competition entries

Usage:
    python -m scripts.paper.generate_figures
"""

import json
import pathlib
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from src.data.loader import build_graph, load_edges
from src.utils.log_utils import setup_logging

matplotlib.use('Agg')

log = setup_logging('generate_figures')

INTERIM     = pathlib.Path('data/interim')
RAW         = pathlib.Path('data/raw')
PREDICTIONS = pathlib.Path('outputs/predictions')
OUT_FIGS    = pathlib.Path('outputs/figures')
PAPER_FIGS  = pathlib.Path('paper/figures')

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
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    src = OUT_FIGS / name
    dst = PAPER_FIGS / name
    fig.savefig(src, dpi=150, bbox_inches='tight')
    shutil.copy2(src, dst)
    log.info('Saved → %s  (copied to %s)', src, dst)
    plt.close(fig)


def _load_json(path: pathlib.Path):
    if not path.exists():
        log.warning('SKIP: %s not found', path)
        return None
    return json.loads(path.read_text())


def _load_csv(path: pathlib.Path):
    if not path.exists():
        log.warning('SKIP: %s not found', path)
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Figure 1: separability distributions
# ---------------------------------------------------------------------------

def fig_separability(subsample: int = 100_000) -> None:
    log.info('Generating separability_distributions.png ...')
    train = pd.read_csv(RAW / 'train.csv')
    structural = pd.read_csv(INTERIM / 'structural_train.csv', index_col='id')
    tfidf = pd.read_csv(INTERIM / 'tfidf_train.csv', index_col='id')

    # Exclude self-loops
    mask = (train['id1'] != train['id2']).values
    labels  = train['label'].values[mask]
    cn      = structural['cn'].values[mask]
    tfidf_s = tfidf['tfidf_score'].values[mask]

    # Subsample for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(labels), min(subsample, len(labels)), replace=False)
    labels, cn, tfidf_s = labels[idx], cn[idx], tfidf_s[idx]

    pos_mask = labels == 1
    neg_mask = labels == 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # CN distribution — log scale (most values are 0, a few are high)
    bins_cn = np.arange(0, min(cn.max() + 2, 20))
    axes[0].hist(cn[pos_mask], bins=bins_cn, alpha=0.7, color=PALETTE['pos'],
                 label='Θετικά', density=True)
    axes[0].hist(cn[neg_mask], bins=bins_cn, alpha=0.7, color=PALETTE['neg'],
                 label='Αρνητικά', density=True)
    axes[0].set_xlabel('Κοινοί Γείτονες (CN)')
    axes[0].set_ylabel('Πυκνότητα')
    axes[0].set_title('Κατανομή Κοινών Γειτόνων')
    axes[0].legend()
    axes[0].set_yscale('log')

    # TF-IDF distribution
    axes[1].hist(tfidf_s[pos_mask], bins=50, alpha=0.7, color=PALETTE['pos'],
                 label='Θετικά', density=True)
    axes[1].hist(tfidf_s[neg_mask], bins=50, alpha=0.7, color=PALETTE['neg'],
                 label='Αρνητικά', density=True)
    axes[1].set_xlabel('Ομοιότητα TF-IDF')
    axes[1].set_ylabel('Πυκνότητα')
    axes[1].set_title('Κατανομή Ομοιότητας TF-IDF')
    axes[1].legend()

    fig.tight_layout()
    _save(fig, 'separability_distributions.png')


# ---------------------------------------------------------------------------
# Figure 2: difficulty breakdown
# ---------------------------------------------------------------------------

def fig_difficulty() -> None:
    log.info('Generating difficulty_breakdown.png ...')
    diff = pd.read_csv(INTERIM / 'difficulty_train.csv', index_col='id')

    counts = diff['difficulty'].value_counts()
    total  = len(diff)

    pcts = [100 * counts.get(cat, 0) / total for cat in DIFF_ORDER]
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

def fig_svm_metrics() -> None:
    log.info('Generating svm_metrics.png ...')
    svm = _load_json(PREDICTIONS / 'svm_val_metrics.json')
    if svm is None:
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

def fig_tier_routing() -> None:
    log.info('Generating tier_routing.png ...')
    vt = _load_csv(PREDICTIONS / 'cascade_val_tiers.csv')
    if vt is None:
        return

    n_total = len(vt)
    counts = vt['tier_used'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    left = 0
    for tier in sorted(TIER_NAMES):
        n = int(counts.get(tier, 0))
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

def fig_tier_difficulty_heatmap() -> None:
    log.info('Generating tier_difficulty_heatmap.png ...')
    vt = _load_csv(PREDICTIONS / 'cascade_val_tiers.csv')
    if vt is None:
        return

    tiers = sorted(vt['tier_used'].unique())
    diffs = [c for c in DIFF_ORDER if c in vt['difficulty'].unique()]

    acc = np.full((len(tiers), len(diffs)), np.nan)
    for i, tier in enumerate(tiers):
        for j, diff in enumerate(diffs):
            sub = vt[(vt['tier_used'] == tier) & (vt['difficulty'] == diff)]
            if len(sub):
                acc[i, j] = sub['correct'].mean()

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

def fig_confusion_matrices() -> None:
    log.info('Generating tier_confusion_matrices.png ...')
    vt = _load_csv(PREDICTIONS / 'cascade_val_tiers.csv')
    if vt is None:
        return

    tiers = sorted(vt['tier_used'].unique())
    fig, axes = plt.subplots(1, len(tiers), figsize=(3.2 * len(tiers), 3.2))
    if len(tiers) == 1:
        axes = [axes]

    for ax, tier in zip(axes, tiers):
        sub = vt[vt['tier_used'] == tier]
        cm = confusion_matrix(sub['y_true'], sub['y_pred'], labels=[0, 1])
        im = ax.imshow(cm, cmap='Blues')
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

def fig_roc_curve() -> None:
    log.info('Generating roc_curve.png ...')
    vt = _load_csv(PREDICTIONS / 'cascade_val_tiers.csv')
    se = _load_csv(PREDICTIONS / 'svm_val_errors.csv')

    fig, ax = plt.subplots(figsize=(6, 5.5))
    plotted = False

    if vt is not None and 'score' in vt.columns:
        fpr, tpr, _ = roc_curve(vt['y_true'], vt['score'])
        ax.plot(fpr, tpr, color=PALETTE['pos'], label='CascadeLP')
        plotted = True
    elif vt is not None:
        log.warning('cascade_val_tiers.csv has no score column (stale run) — skipping CascadeLP ROC')

    if se is not None and 'score' in se.columns:
        fpr, tpr, _ = roc_curve(se['y_true'], se['score'])
        ax.plot(fpr, tpr, color=PALETTE['neg'], label='SVM (TF-IDF)')
        plotted = True
    elif se is not None:
        log.warning('svm_val_errors.csv has no score column (stale run) — skipping SVM ROC')

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

def fig_confidence_distribution() -> None:
    log.info('Generating tier_confidence_distribution.png ...')
    vt = _load_csv(PREDICTIONS / 'cascade_val_tiers.csv')
    if vt is None or 'score' not in vt.columns:
        log.warning('SKIP: cascade_val_tiers.csv missing or has no score column')
        return

    tiers = [t for t in sorted(vt['tier_used'].unique()) if t > 0]
    fig, axes = plt.subplots(1, len(tiers), figsize=(3.5 * len(tiers), 3.2), sharey=True)
    if len(tiers) == 1:
        axes = [axes]

    for ax, tier in zip(axes, tiers):
        sub = vt[vt['tier_used'] == tier]
        ax.hist(sub['score'], bins=30, color=PALETTE['colors'][tier % len(PALETTE['colors'])])
        ax.set_title(TIER_NAMES.get(tier, str(tier)).replace('\n', ' '), fontsize=9)
        ax.set_xlabel('P(y=1)')
    axes[0].set_ylabel('Πλήθος ζευγών')

    fig.suptitle('Κατανομή Βαθμολογίας Εμπιστοσύνης ανά Επίπεδο')
    fig.tight_layout()
    _save(fig, 'tier_confidence_distribution.png')


# ---------------------------------------------------------------------------
# Figure 9: threshold ablation curves
# ---------------------------------------------------------------------------

def fig_threshold_ablation() -> None:
    log.info('Generating threshold_ablation.png ...')
    abl = _load_csv(PREDICTIONS / 'cascade_threshold_ablation.csv')
    if abl is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for i, t2 in enumerate(sorted(abl['tau2'].unique())):
        sub = abl[abl['tau2'] == t2].sort_values('tau1')
        axes[0].plot(sub['tau1'], sub['macro_f1'], marker='o',
                     color=PALETTE['colors'][i % len(PALETTE['colors'])], label=f'$\\tau_2$={t2}')
    axes[0].set_xlabel('$\\tau_1$')
    axes[0].set_ylabel('Macro F1')
    axes[0].set_title('Macro F1 vs. $\\tau_1$')
    axes[0].legend(fontsize=8)

    t1_08 = abl[abl['tau1'] == 0.8].sort_values('tau2')
    axes[1].plot(t1_08['tau2'], t1_08['tier3_pct'], marker='o', color=PALETTE['neg'])
    axes[1].set_xlabel('$\\tau_2$')
    axes[1].set_ylabel('Ποσοστό κλήσεων Επιπέδου 3 (%)')
    axes[1].set_title('Ρυθμός Κλήσης Επιπέδου 3 vs. $\\tau_2$ ($\\tau_1$=0.8)')

    fig.tight_layout()
    _save(fig, 'threshold_ablation.png')


# ---------------------------------------------------------------------------
# Figure 10: Node2Vec ablation comparison
# ---------------------------------------------------------------------------

def fig_node2vec_ablation() -> None:
    log.info('Generating node2vec_ablation.png ...')
    n2v = _load_json(PREDICTIONS / 'node2vec_ablation.json')
    if n2v is None:
        return

    without, with_n2v = n2v['without'], n2v['with']
    metrics = ['accuracy', 'macro_f1', 'accuracy_confident', 'macro_f1_confident']
    labels = ['Ακρίβεια\n(σύνολο)', 'Macro F1\n(σύνολο)', 'Ακρίβεια\n(σίγουρα)', 'Macro F1\n(σίγουρα)']

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, [without[m] for m in metrics], width, label='Χωρίς Node2Vec', color=PALETTE['colors'][0])
    ax.bar(x + width / 2, [with_n2v[m] for m in metrics], width, label='Με Node2Vec', color=PALETTE['colors'][1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Τιμή')
    ax.set_title('Επίδραση Node2Vec στο Επίπεδο 1 (Validation)')
    ax.legend()
    fig.tight_layout()
    _save(fig, 'node2vec_ablation.png')


# ---------------------------------------------------------------------------
# Figure 11: throughput comparison, CascadeLP vs. SVM
# ---------------------------------------------------------------------------

def fig_throughput_comparison() -> None:
    log.info('Generating throughput_comparison.png ...')
    thr = _load_json(PREDICTIONS / 'throughput_benchmark.json')
    svm = _load_json(PREDICTIONS / 'svm_val_metrics.json')
    if thr is None or svm is None:
        log.warning('SKIP: need both throughput_benchmark.json and svm_val_metrics.json')
        return

    svm_ms_per_pair = svm['latency_ms'] / svm['n_val'] if svm.get('latency_ms') and svm.get('n_val') else None
    if svm_ms_per_pair is None:
        log.warning('SKIP: svm_val_metrics.json missing latency_ms/n_val')
        return

    names = ['CascadeLP', 'SVM (TF-IDF)']
    values = [thr['ms_per_pair'], svm_ms_per_pair]

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

def fig_coldstart_comparison() -> None:
    log.info('Generating coldstart_comparison.png ...')
    cascade = _load_json(PREDICTIONS / 'cascade_val_metrics.json')
    svm = _load_json(PREDICTIONS / 'svm_val_metrics.json')
    if cascade is None or svm is None:
        log.warning('SKIP: need both cascade_val_metrics.json and svm_val_metrics.json')
        return

    names = ['CascadeLP', 'SVM (TF-IDF)']
    overall = [cascade['macro_f1'], svm['macro_f1']]
    cold_start = [cascade['cold_start_f1'], svm['cold_start_f1']]

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

def fig_error_by_difficulty_comparison() -> None:
    log.info('Generating error_by_difficulty_comparison.png ...')
    vt = _load_csv(PREDICTIONS / 'cascade_val_tiers.csv')
    se = _load_csv(PREDICTIONS / 'svm_val_errors.csv')
    if vt is None or se is None:
        log.warning('SKIP: need both cascade_val_tiers.csv and svm_val_errors.csv')
        return

    diffs = [c for c in DIFF_ORDER if c in vt['difficulty'].unique()]
    cascade_err_pct = [100 * (1 - vt[vt['difficulty'] == d]['correct'].mean()) for d in diffs]
    svm_err_pct = [100 * (1 - se[se['difficulty'] == d]['correct'].mean()) if d in se['difficulty'].unique() else 0
                   for d in diffs]

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

def fig_graph_degree_distribution() -> None:
    log.info('Generating graph_degree_distribution.png ...')
    train = load_edges(f'{RAW}/train.csv')
    G = build_graph(train)
    degrees = np.array([d for _, d in G.degree()])

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(0, np.log10(max(degrees.max(), 2)), 40)
    ax.hist(degrees, bins=bins, color=PALETTE['colors'][0])
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

def fig_dataset_composition() -> None:
    log.info('Generating dataset_composition.png ...')
    train = pd.read_csv(RAW / 'train.csv')
    diff = pd.read_csv(INTERIM / 'difficulty_train.csv', index_col='id')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    label_counts = train['label'].value_counts()
    axes[0].bar(['Αρνητικά', 'Θετικά'], [label_counts.get(0, 0), label_counts.get(1, 0)],
                color=[PALETTE['neg'], PALETTE['pos']], width=0.5)
    axes[0].set_ylabel('Πλήθος Ζευγών')
    axes[0].set_title('Ισορροπία Κλάσεων — train.csv')
    for i, v in enumerate([label_counts.get(0, 0), label_counts.get(1, 0)]):
        axes[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)

    counts = diff['difficulty'].value_counts()
    axes[1].bar([DIFF_LABELS[c] for c in DIFF_ORDER], [counts.get(c, 0) for c in DIFF_ORDER],
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
# paper/body_matter/chap2.tex's tab:dsaa-submissions exactly (external, citation-backed
# facts — not this project's experiment output, hence literal here per CLAUDE.md's
# no-magic-numbers carve-out). Teams with no reported numeric score in their short paper
# (nguyen2023mat, mata2023link, kansal2023predict) are omitted from the chart below.
_DSAA_OTHER_TEAMS = [
    ('UIT-NLP\n(phan2023link)', 1.0),
    ('Tran et al.\n(tran2023text)', 0.99999),
    ('Yang\n(yang2023achieving)', 0.99),
    ('Giannoulidis &\nMavroudopoulos', 0.948),
]


def fig_dsaa_leaderboard_comparison() -> None:
    log.info('Generating dsaa_leaderboard_comparison.png ...')
    kg = _load_csv(PREDICTIONS / 'kaggle_scores.csv')
    if kg is None:
        return
    ref_rows = kg[~kg['description'].str.contains('Node2Vec', na=False)]
    if ref_rows.empty:
        log.warning('SKIP: no non-Node2Vec Kaggle submission found for our own score')
        return
    our_score = ref_rows.sort_values('public_score', ascending=False).iloc[0]['private_score']

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
    fig_separability()
    fig_difficulty()
    fig_svm_metrics()
    fig_tier_routing()
    fig_tier_difficulty_heatmap()
    fig_confusion_matrices()
    fig_roc_curve()
    fig_confidence_distribution()
    fig_threshold_ablation()
    fig_node2vec_ablation()
    fig_throughput_comparison()
    fig_coldstart_comparison()
    fig_error_by_difficulty_comparison()
    fig_graph_degree_distribution()
    fig_dataset_composition()
    fig_dsaa_leaderboard_comparison()
    log.info('Done.')


if __name__ == '__main__':
    main()
