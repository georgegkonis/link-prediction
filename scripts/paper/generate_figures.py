"""
Generate thesis figures from experiment data and save to outputs/figures/ and paper/figures/.

Produces:
  separability_distributions.png  — CN and TF-IDF distributions for pos vs. neg pairs
  difficulty_breakdown.png        — difficulty-category bar chart (train set)
  svm_metrics.png                 — SVM baseline performance bar chart

Usage:
    python -m scripts.generate_figures
"""

import json
import pathlib
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    'colors': ['#2196F3', '#4CAF50', '#FF9800', '#F44336'],
}


def _save(fig: plt.Figure, name: str) -> None:
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    src = OUT_FIGS / name
    dst = PAPER_FIGS / name
    fig.savefig(src, dpi=150, bbox_inches='tight')
    shutil.copy2(src, dst)
    log.info('Saved → %s  (copied to %s)', src, dst)
    plt.close(fig)


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

    labels_map = {
        'trivial_self_loop':    'Αυτοβρόχοι',
        'trivial_high_cn':      'Υψηλό CN',
        'trivial_high_textsim': 'Υψηλή Ομοιότητα',
        'hard':                 'Δύσκολα',
    }
    order = ['trivial_self_loop', 'trivial_high_cn', 'trivial_high_textsim', 'hard']
    counts = diff['difficulty'].value_counts()
    total  = len(diff)

    pcts = [100 * counts.get(cat, 0) / total for cat in order]
    names = [labels_map[cat] for cat in order]

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
    svm_path = PREDICTIONS / 'svm_val_metrics.json'
    if not svm_path.exists():
        log.warning('SKIP: svm_val_metrics.json not found — run make train MODEL=svm first')
        return

    svm = json.loads(svm_path.read_text())
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fig_separability()
    fig_difficulty()
    fig_svm_metrics()
    log.info('Done.')


if __name__ == '__main__':
    main()
