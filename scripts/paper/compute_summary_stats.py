"""
Compute every statistic, metric, and aggregate needed by generate_macros.py and
generate_figures.py, and persist them to outputs/stats/summary_stats.json.

This is the ONLY script in the paper-asset pipeline that touches data/raw/dsaa/, data/interim/dsaa/, or the
per-pair CSVs in outputs/predictions/dsaa/ — all of which are large, regenerable pipeline artifacts and
stay .gitignore'd. Everything downstream (generate_macros.py, generate_figures.py) reads only the
small committed summary_stats.json, so figures/macros can be regenerated from a clean checkout
without re-running the data/feature/train pipeline.

Reads:
  data/raw/dsaa/train.csv, test.csv, nodes.tsv              — dataset sizes, graph stats
  data/interim/dsaa/difficulty_train.csv                    — difficulty breakdown (train)
  data/interim/dsaa/difficulty_test.csv                     — difficulty breakdown (test)
  data/interim/dsaa/difficulty_thresholds.json              — CN / TF-IDF thresholds
  data/interim/dsaa/leakage_pairs.csv                       — leakage audit
  data/interim/dsaa/structural_train.csv                    — CN / graph coverage
  data/interim/dsaa/tfidf_train.csv                         — TF-IDF separability
  outputs/predictions/dsaa/cascade_val_tiers.csv            — per-pair val predictions (+ scores)
  outputs/predictions/dsaa/cascade_test_tiers.csv           — per-pair test predictions
  outputs/predictions/dsaa/cascade_n2v_test_tiers.csv       — per-pair test predictions (n2v variant)
  outputs/predictions/dsaa/cascade_threshold_ablation.csv   — threshold sweep
  outputs/predictions/dsaa/cascade_val_metrics.json         — cascade scalar val metrics
  outputs/predictions/dsaa/kaggle_scores.csv                — Kaggle leaderboard scores
  outputs/predictions/dsaa/svm_val_metrics.json             — SVM scalar metrics
  outputs/predictions/dsaa/svm_val_errors.csv               — SVM per-pair val errors by difficulty
  outputs/predictions/dsaa/node2vec_ablation.json           — Node2Vec with/without ablation + coverage
  outputs/predictions/dsaa/hard_residual_analysis.json      — Tier-3 hard-residual / nodes.tsv join
  outputs/predictions/dsaa/throughput_benchmark.json        — CPU inference throughput
  outputs/predictions/dsaa/tier2_confidence_saturation.json — Tier-2 RandomForest confidence ceiling
  outputs/predictions/dsaa/embedding_val_metrics.json       — standalone EmbeddingClassifier baseline

Writes:
  outputs/stats/summary_stats.json

Usage:
    python -m scripts.paper.compute_summary_stats
"""

import json
import pathlib

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.data.loader import build_graph
from src.utils.log_utils import setup_logging

log = setup_logging('compute_summary_stats')

INTERIM     = pathlib.Path('data/interim/dsaa')
RAW         = pathlib.Path('data/raw/dsaa')
PREDICTIONS = pathlib.Path('outputs/predictions/dsaa')
STATS       = pathlib.Path('outputs/stats')
CONFIGS     = pathlib.Path('configs')

_MODEL_NAMES = ('structural', 'tfidf', 'pos', 'embedding', 'svm', 'cascade')


def _load_configs() -> dict:
    """
    Read configs/*.yaml directly (plain YAML, no Hydra runtime needed here — we
    only need the resolved literal values). This is the single source of truth
    for every hyperparameter/reproducibility macro below; nothing here should
    ever be a hardcoded Python literal that could drift from configs/.
    """
    cfg = yaml.safe_load((CONFIGS / 'config.yaml').read_text())
    cfg['training'] = yaml.safe_load((CONFIGS / 'training/default.yaml').read_text())
    cfg['features'] = yaml.safe_load((CONFIGS / 'features/default.yaml').read_text())
    cfg['model'] = {
        name: yaml.safe_load((CONFIGS / f'model/{name}.yaml').read_text())
        for name in _MODEL_NAMES
    }
    return cfg

_MISSING = '---'

DIFF_ORDER = ['trivial_self_loop', 'trivial_high_cn', 'trivial_high_textsim', 'hard']

# ---------------------------------------------------------------------------
# Greek number formatters (macros are pre-formatted here — the only consumer,
# generate_macros.py, just writes them out verbatim as \newcommand values)
# ---------------------------------------------------------------------------

def gint(n: int) -> str:
    """948232 → '948.232' (Greek thousands separator)."""
    return f'{int(n):,}'.replace(',', '.')


def gfloat(x: float, d: int = 4) -> str:
    """0.9986 → '0{,}9986' (Greek decimal comma)."""
    s = f'{float(x):.{d}f}'
    i, f = s.split('.')
    i_fmt = f'{int(i):,}'.replace(',', '.')
    return f'{i_fmt}{{,}}{f}'


def gpct(x: float, d: int = 2) -> str:
    """19.38 → '19{,}38'."""
    s = f'{float(x):.{d}f}'
    i, f = s.split('.')
    return f'{i}{{,}}{f}'


def _count_lines(path: pathlib.Path) -> int:
    """Fast row count (minus header) without a full pandas parse of a 641MB file."""
    with open(path, 'rb') as f:
        return sum(1 for _ in f) - 1


def _load_json_optional(path: pathlib.Path, hint: str) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    log.warning('%s not found — %s', path, hint)
    return None


def _load_csv_optional(path: pathlib.Path, hint: str) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    log.warning('%s not found — %s', path, hint)
    return None


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load() -> dict:
    log.info('Loading CSVs...')
    d: dict = {}

    d['cfg'] = _load_configs()

    d['train'] = pd.read_csv(RAW / 'train.csv')
    d['test']  = pd.read_csv(RAW / 'test.csv')

    d['diff_train'] = pd.read_csv(INTERIM / 'difficulty_train.csv', index_col='id')
    d['diff_test']  = pd.read_csv(INTERIM / 'difficulty_test.csv',  index_col='id')

    thresh_path = INTERIM / 'difficulty_thresholds.json'
    if thresh_path.exists():
        d['thresholds'] = json.loads(thresh_path.read_text())
    else:
        log.warning('difficulty_thresholds.json not found — run make analyze-dataset first')
        d['thresholds'] = {'cn_threshold': 0.0, 'tfidf_threshold': 0.098}

    d['struct_train'] = pd.read_csv(INTERIM / 'structural_train.csv', index_col='id')
    d['tfidf_train']  = pd.read_csv(INTERIM / 'tfidf_train.csv',      index_col='id')

    d['leakage']    = pd.read_csv(INTERIM / 'leakage_pairs.csv')
    d['val_tiers']  = pd.read_csv(PREDICTIONS / 'cascade_val_tiers.csv')
    d['test_tiers'] = pd.read_csv(PREDICTIONS / 'cascade_test_tiers.csv')
    d['ablation']   = pd.read_csv(PREDICTIONS / 'cascade_threshold_ablation.csv')
    d['kaggle']     = pd.read_csv(PREDICTIONS / 'kaggle_scores.csv')

    d['svm'] = _load_json_optional(
        PREDICTIONS / 'svm_val_metrics.json', 'run make train MODEL=svm first')
    d['svm_errors'] = _load_csv_optional(
        PREDICTIONS / 'svm_val_errors.csv', 'run make train MODEL=svm first (writes per-pair errors too)')
    d['n2v_ablation'] = _load_json_optional(
        PREDICTIONS / 'node2vec_ablation.json', 'run python -m scripts.analysis.ablate_node2vec first')
    d['hard_residual'] = _load_json_optional(
        PREDICTIONS / 'hard_residual_analysis.json',
        'run python -m scripts.analysis.analyze_hard_residual first')
    d['throughput'] = _load_json_optional(
        PREDICTIONS / 'throughput_benchmark.json',
        'run python -m scripts.analysis.benchmark_throughput first')
    d['tier2_saturation'] = _load_json_optional(
        PREDICTIONS / 'tier2_confidence_saturation.json',
        'run python -m scripts.analysis.ablate_cascade_thresholds first')
    d['embedding'] = _load_json_optional(
        PREDICTIONS / 'embedding_val_metrics.json', 'run make train MODEL=embedding first')
    d['n2v_test_tiers'] = _load_csv_optional(
        PREDICTIONS / 'cascade_n2v_test_tiers.csv',
        'run python -m scripts.evaluate model=cascade training.no_n2v=false +tag=n2v first')
    d['cascade_val_metrics'] = _load_json_optional(
        PREDICTIONS / 'cascade_val_metrics.json', 'run make train MODEL=cascade first')

    # ---- Negative-sampling artifact audit (explains the near-perfect DSAA 2023 score) ----
    d['neg_sampling_audit'] = _load_json_optional(
        STATS / 'negative_sampling_audit.json', 'run python -m scripts.analysis.audit_negative_sampling first')
    d['hub_in_predictions'] = _load_json_optional(
        STATS / 'hub_in_predictions_audit.json', 'run python -m scripts.analysis.audit_hub_in_predictions first')

    # ---- Fresh, artifact-free Wikipedia validation dataset ----
    d['wikipedia'] = _load_json_optional(
        STATS / 'wiki_cs_8k_experiment_results.json',
        'run python -m scripts.analysis.run_wiki_cs_8k_experiment first')
    d['wikipedia_crawl'] = _load_json_optional(
        pathlib.Path('data/raw/wiki_cs_8k/crawl_stats.json'),
        'run python -m scripts.data.build_from_wikidump first')
    d['wikipedia_text'] = _load_json_optional(
        pathlib.Path('data/raw/wiki_cs_8k/text_fetch_stats.json'),
        'run python -m scripts.data.fetch_wikipedia_text first')

    return d


# ---------------------------------------------------------------------------
# Compute macros (Greek-formatted tex values)
# ---------------------------------------------------------------------------

def _compute_macros(d: dict) -> tuple[dict[str, str], dict]:
    """Returns (macros, shared) — shared holds raw intermediates figures also need."""
    m: dict[str, str] = {}
    shared: dict = {}

    cfg = d['cfg']
    seed     = cfg['seed']
    val_size = cfg['training']['val_size']

    # ---- Dataset sizes ----
    n_train_raw   = len(d['train'])
    n_test        = len(d['test'])
    self_loop_mask = d['train']['id1'] == d['train']['id2']
    n_self_loops  = self_loop_mask.sum()
    n_no_self     = n_train_raw - n_self_loops

    y = d['train']['label'].values
    not_self_idx = np.where(~self_loop_mask.values)[0]
    tr, val = train_test_split(not_self_idx, test_size=val_size, stratify=y[not_self_idx], random_state=seed)
    n_tr, n_val = len(tr), len(val)

    # Graph stats — live, from the built positive-edge graph and a fast nodes.tsv line count
    G = build_graph(d['train'])
    graph_nodes_structural = G.number_of_nodes()
    graph_mean_degree = float(np.mean([deg for _, deg in G.degree()])) if graph_nodes_structural else 0.0
    nodes_path = RAW / 'nodes.tsv'
    if nodes_path.exists():
        graph_nodes_total = _count_lines(nodes_path)
    else:
        log.warning('%s not found — GraphNodesTotal falling back to structural node count', nodes_path)
        graph_nodes_total = graph_nodes_structural

    m['TrainPairs']       = gint(n_train_raw)
    m['TrainPairsNoSelf'] = gint(n_no_self)
    m['TrainSplitSize']   = gint(n_tr)
    m['ValSplitSize']     = gint(n_val)
    m['TestPairs']        = gint(n_test)
    m['GraphNodes']       = gint(graph_nodes_structural)
    m['GraphMeanDegree']  = gfloat(graph_mean_degree, 2)
    m['GraphNodesTotal']  = gint(graph_nodes_total)

    # ---- Reproducibility & hyperparameters (sourced from configs/, single source of truth) ----
    m['Seed']       = str(seed)
    m['ValSizeFrac'] = gfloat(val_size, 1)
    m['ValSizePct'] = gpct(100 * val_size, 1)

    struct_cfg, tfidf_cfg, pos_cfg = cfg['model']['structural'], cfg['model']['tfidf'], cfg['model']['pos']
    emb_cfg, svm_cfg, cascade_cfg  = cfg['model']['embedding'], cfg['model']['svm'], cfg['model']['cascade']

    m['StructuralC']       = gfloat(struct_cfg['C'], 1)
    m['StructuralMaxIter'] = gint(struct_cfg['max_iter'])
    m['TfidfClfC']         = gfloat(tfidf_cfg['C'], 1)
    m['TfidfClfMaxIter']   = gint(tfidf_cfg['max_iter'])
    m['PosNEstimators']    = gint(pos_cfg['n_estimators'])
    m['PosClassWeight']    = str(pos_cfg['class_weight'])
    m['EmbeddingC']        = gfloat(emb_cfg['C'], 1)
    m['EmbeddingMaxIter']  = gint(emb_cfg['max_iter'])
    m['SvmC']              = gfloat(svm_cfg['C'], 1)
    m['SvmGamma']          = str(svm_cfg['gamma'])
    m['SvmSampleSize']     = gint(svm_cfg['subsample_size'])
    m['TauOneDefault']     = gfloat(cascade_cfg['tier1_threshold'], 1)
    m['TauTwoDefault']     = gfloat(cascade_cfg['tier2_threshold'], 1)

    tfidf_feat_cfg = cfg['features']['tfidf']
    m['TfidfMaxFeatures'] = gint(tfidf_feat_cfg['max_features'])
    m['TfidfSublinearTf'] = str(tfidf_feat_cfg['sublinear_tf'])
    m['TfidfMinDf']       = gint(tfidf_feat_cfg['min_df'])
    m['EmbeddingModelName'] = str(cfg['features']['embedding']['model_name'])

    n2v_cfg = cfg['features']['node2vec']
    m['NTwoVDimensions'] = gint(n2v_cfg['dimensions'])
    m['NTwoVWalkLength'] = gint(n2v_cfg['walk_length'])
    m['NTwoVNumWalks']   = gint(n2v_cfg['num_walks'])
    m['NTwoVWindow']     = gint(n2v_cfg['window'])
    m['NTwoVP']          = str(int(n2v_cfg['p']))
    m['NTwoVQ']          = str(int(n2v_cfg['q']))
    # Tier-1 feature vector width when Node2Vec is enabled: 4 structural heuristics
    # concatenated with the Node2Vec Hadamard embedding — derived, not duplicated.
    m['NTwoVDim']       = gint(n2v_cfg['dimensions'] + 4)

    # Class balance (full train.csv including self-loops)
    n_train_pos = int((d['train']['label'] == 1).sum())
    n_train_neg = int((d['train']['label'] == 0).sum())
    m['TrainPosCount'] = gint(n_train_pos)
    m['TrainNegCount'] = gint(n_train_neg)
    m['TrainPosPct']   = gpct(100 * n_train_pos / n_train_raw, 1)
    m['TrainNegPct']   = gpct(100 * n_train_neg / n_train_raw, 1)

    # ---- Leakage audit ----
    exact    = (d['leakage']['kind'] == 'exact').sum()
    reversed_ = (d['leakage']['kind'] == 'reversed').sum()
    self_pos  = (d['train'].loc[self_loop_mask, 'label'] == 1).sum()
    self_neg  = (d['train'].loc[self_loop_mask, 'label'] == 0).sum()

    # self-loops in test = diff_test trivial_self_loop count
    test_self = (d['diff_test']['difficulty'] == 'trivial_self_loop').sum()
    # intra-train duplicates: pairs appearing >1 time (undirected)
    pairs_sorted = d['train'][~self_loop_mask][['id1', 'id2']].apply(
        lambda r: tuple(sorted([r['id1'], r['id2']])), axis=1
    )
    dup_groups = pairs_sorted.value_counts()
    n_dup_groups = (dup_groups > 1).sum()

    m['LeakageExact']          = str(exact)
    m['LeakageExactPct']       = gfloat(100 * exact / n_test, 2)
    m['LeakageReversed']       = gint(reversed_)
    m['LeakageReversedPct']    = gfloat(100 * reversed_ / n_test, 2)
    m['SelfLoopsTrain']        = gint(n_self_loops)
    m['SelfLoopsTrainPos']     = gint(self_pos)
    m['SelfLoopsTrainNeg']     = str(int(self_neg))
    m['SelfLoopsTest']         = gint(test_self)
    m['IntraTrainDuplicates']  = gint(n_dup_groups)
    m['IntraTrainDuplicatesRows'] = gint(n_dup_groups * 2)

    # ---- Separability thresholds ----
    cn_thr    = d['thresholds']['cn_threshold']
    tfidf_thr = d['thresholds']['tfidf_threshold']
    m['CnThreshold']    = gfloat(cn_thr, 3)
    m['TfidfThreshold'] = gfloat(tfidf_thr, 3)

    # ---- Separability distributions (CN / TF-IDF means, positive vs. negative) ----
    train_ids  = d['train']['id'].values
    cn_vals    = d['struct_train']['cn'].reindex(train_ids).values
    tfidf_vals = d['tfidf_train']['tfidf_score'].reindex(train_ids).values
    not_self   = ~self_loop_mask.values

    m['StructCnMeanPos']    = gfloat(np.nanmean(cn_vals[not_self & (y == 1)]), 5)
    m['StructCnMeanNeg']    = gfloat(np.nanmean(cn_vals[not_self & (y == 0)]), 5)
    m['StructTfidfMeanPos'] = gfloat(np.nanmean(tfidf_vals[not_self & (y == 1)]), 3)
    m['StructTfidfMeanNeg'] = gfloat(np.nanmean(tfidf_vals[not_self & (y == 0)]), 3)

    # ---- Structural graph coverage (both endpoints present in the positive-edge graph) ----
    graph_nodes = set(G.nodes())
    train_nonself = d['train'][~self_loop_mask]
    m['GraphCoverageTrainPct'] = gpct(100 * (
        train_nonself['id1'].isin(graph_nodes) & train_nonself['id2'].isin(graph_nodes)
    ).mean(), 2)
    m['GraphCoverageTestPct'] = gpct(100 * (
        d['test']['id1'].isin(graph_nodes) & d['test']['id2'].isin(graph_nodes)
    ).mean(), 2)

    # ---- Difficulty distribution (train) ----
    diff_train_counts = d['diff_train']['difficulty'].value_counts()
    n_dt = len(d['diff_train'])

    def _diff_n(cat):
        return diff_train_counts.get(cat, 0)
    def _diff_pct(cat):
        return 100 * _diff_n(cat) / n_dt

    m['TrainDiffSelfLoopN']      = gint(_diff_n('trivial_self_loop'))
    m['TrainDiffSelfLoopPct']    = gpct(_diff_pct('trivial_self_loop'))
    m['TrainDiffHighCnN']        = gint(_diff_n('trivial_high_cn'))
    m['TrainDiffHighCnPct']      = gpct(_diff_pct('trivial_high_cn'))
    m['TrainDiffHighTextsimN']   = gint(_diff_n('trivial_high_textsim'))
    m['TrainDiffHighTextsimPct'] = gpct(_diff_pct('trivial_high_textsim'))
    m['TrainDiffHardN']          = gint(_diff_n('hard'))
    m['TrainDiffHardPct']        = gpct(_diff_pct('hard'))
    trivial_pct = 100 - _diff_pct('hard')
    m['TrainDiffTrivialPct']     = gpct(trivial_pct)

    # ---- Difficulty distribution (test) ----
    diff_test_counts = d['diff_test']['difficulty'].value_counts()
    n_dtest = len(d['diff_test'])

    def _tdiff_n(cat):
        return diff_test_counts.get(cat, 0)
    def _tdiff_pct(cat):
        return 100 * _tdiff_n(cat) / n_dtest

    m['TestDiffSelfLoopN']      = gint(_tdiff_n('trivial_self_loop'))
    m['TestDiffSelfLoopPct']    = gpct(_tdiff_pct('trivial_self_loop'))
    m['TestDiffHighCnN']        = gint(_tdiff_n('trivial_high_cn'))
    m['TestDiffHighTextsimN']   = gint(_tdiff_n('trivial_high_textsim'))
    m['TestDiffHighTextsimPct'] = gpct(_tdiff_pct('trivial_high_textsim'))
    m['TestDiffHardN']          = gint(_tdiff_n('hard'))
    m['TestDiffHardPct']        = gpct(_tdiff_pct('hard'))

    # ---- Cascade validation results ----
    vt = d['val_tiers']
    n_vt = len(vt)

    def _tier_stats(tier_id):
        sub = vt[vt['tier_used'] == tier_id]
        n   = len(sub)
        if n == 0:
            return n, 0.0, float('nan'), float('nan')
        acc  = sub['correct'].mean()
        f1   = f1_score(sub['y_true'], sub['y_pred'], average='macro', zero_division=0)
        return n, 100 * n / n_vt, acc, f1

    t1_n, t1_pct, t1_acc, t1_f1 = _tier_stats(1)
    t2_n, t2_pct, t2_acc, t2_f1 = _tier_stats(2)
    t3_n, t3_pct, t3_acc, t3_f1 = _tier_stats(3)

    cascade_f1  = f1_score(vt['y_true'], vt['y_pred'], average='macro', zero_division=0)
    if 'score' in vt.columns:
        cascade_auc = roc_auc_score(vt['y_true'], vt['score'])
    else:
        log.warning("cascade_val_tiers.csv has no 'score' column (stale run) — "
                    'AUC falling back to discrete predictions as a proxy')
        cascade_auc = roc_auc_score(vt['y_true'], vt['y_pred'])

    m['CascadeValFone'] = gfloat(cascade_f1,  4)
    m['CascadeValAuc']  = gfloat(cascade_auc, 4)

    m['TierOneCount'] = gint(t1_n)
    m['TierOnePct']   = gpct(t1_pct)
    m['TierOneAcc']   = gfloat(t1_acc, 4)
    m['TierOneFone']  = gfloat(t1_f1,  4)

    m['TierTwoCount'] = gint(t2_n)
    m['TierTwoPct']   = gpct(t2_pct)
    m['TierTwoAcc']   = gfloat(t2_acc, 4)
    m['TierTwoFone']  = gfloat(t2_f1,  4)

    m['TierThreeCount'] = gint(t3_n)
    m['TierThreePct']   = gpct(t3_pct, 2)
    m['TierThreeAcc']   = gfloat(t3_acc, 4)
    m['TierThreeFone']  = gfloat(t3_f1,  4)

    # Tier 1: what fraction of pairs it ACCEPTS are positive
    t1_sub = vt[vt['tier_used'] == 1]
    t1_pos_pct = 100 * t1_sub['y_true'].mean()
    m['TierOnePosPct'] = gpct(t1_pos_pct, 2)

    # Tier 3: true vs predicted positive rates
    t3_sub = vt[vt['tier_used'] == 3]
    if len(t3_sub):
        m['TierThreeTruePosPct'] = gpct(100 * t3_sub['y_true'].mean(), 1)
        m['TierThreePredPosPct'] = gpct(100 * t3_sub['y_pred'].mean(), 1)
    else:
        m['TierThreeTruePosPct'] = '0{,}0'
        m['TierThreePredPosPct'] = '0{,}0'

    # ---- Tier × difficulty cross-tab (macro subset: hard / high-textsim / high-cn only) ----
    for tier_id, tier_name in [(1, 'One'), (2, 'Two'), (3, 'Three')]:
        for diff_cat, diff_name in [('hard', 'Hard'), ('trivial_high_textsim', 'HighTextsim'),
                                     ('trivial_high_cn', 'HighCn')]:
            sub = vt[(vt['tier_used'] == tier_id) & (vt['difficulty'] == diff_cat)]
            n   = len(sub)
            if n > 0:
                acc = sub['correct'].mean()
                m[f'Tier{tier_name}{diff_name}N']   = gint(n)
                m[f'Tier{tier_name}{diff_name}Acc'] = gfloat(acc, 4)
            else:
                m[f'Tier{tier_name}{diff_name}N']   = '0'
                m[f'Tier{tier_name}{diff_name}Acc'] = _MISSING

    # ---- Cold-start (no common neighbours) ----
    # proxy: pairs routed to Tier 2 or 3 never got a confident Tier-1 structural
    # decision, i.e. they behave as cold-start w.r.t. the structural signal.
    cs_count  = (vt['tier_used'] >= 2).sum()
    cs_pct    = 100 * cs_count / n_vt
    non_cs    = n_vt - cs_count
    m['ColdStartCount']    = gint(cs_count)
    m['ColdStartPct']      = gfloat(cs_pct, 2)
    m['ColdStartNonCount'] = gint(non_cs)

    # ---- Hard residual (Tier 3 ∩ hard) ----
    hard_res = vt[(vt['tier_used'] == 3) & (vt['difficulty'] == 'hard')]
    n_hr     = len(hard_res)
    hr_acc   = hard_res['correct'].mean() if n_hr > 0 else float('nan')

    m['HardResCount'] = gint(n_hr)
    m['HardResAcc']   = gfloat(hr_acc, 4) if n_hr > 0 else _MISSING

    hres = d['hard_residual']
    if hres:
        m['HardResMissingN']      = gint(hres['missing_n'])
        m['HardResMissingPct']    = gpct(hres['missing_pct'], 1)
        m['HardResCompleteN']     = gint(hres['complete_n'])
        m['HardResCompleteAcc']   = gfloat(hres['complete_acc'], 4)
        m['HardResIncompleteAcc'] = gfloat(hres['incomplete_acc'], 4)
        m['HardResNodeId']        = str(hres['top_node_id'])
        m['HardResNodePairs']     = gint(hres['top_node_pairs'])
        m['HardResNodePredPosPct'] = gpct(hres['top_node_pred_pos_pct'], 1)
    else:
        for k in ('HardResMissingN', 'HardResMissingPct', 'HardResCompleteN', 'HardResCompleteAcc',
                  'HardResIncompleteAcc', 'HardResNodeId', 'HardResNodePairs', 'HardResNodePredPosPct'):
            m[k] = _MISSING

    # ---- Test-set tier distribution ----
    # Live for all three tiers once cascade_test_tiers.csv comes from the
    # heuristics-only (training.no_n2v=true) checkpoint — see plan.md's
    # "Code↔thesis mismatch" item. If the on-disk file still looks Node2Vec-
    # contaminated (implausibly high Tier-1 share), warn instead of silently
    # emitting a misleading number.
    tt = d['test_tiers']
    n_tt = len(tt)
    tt0_n = (tt['tier_used'] == 0).sum()
    tt1_n = (tt['tier_used'] == 1).sum()
    tt2_n = (tt['tier_used'] == 2).sum()
    tt1_pct = 100 * tt1_n / n_tt
    if tt1_pct > 50:
        log.warning('cascade_test_tiers.csv Tier-1 share is %.1f%% — this looks like the '
                    'Node2Vec-contaminated checkpoint, not the heuristics-only config. '
                    'Regenerate with training.no_n2v=true before trusting TestTierOne*/TestTierTwoPct.',
                    tt1_pct)
    m['TestTierOneCount'] = gint(tt1_n)
    m['TestTierOnePct']   = gpct(tt1_pct, 1)
    m['TestTierZeroPct']  = gpct(100 * tt0_n / n_tt, 1)
    m['TestTierTwoPct']   = gpct(100 * tt2_n / n_tt, 1)

    # ---- Efficiency / throughput ----
    thr = d['throughput']
    if thr:
        m['ThroughputSec']     = gfloat(thr['wall_clock_sec'], 2)
        m['ThroughputRate']    = gint(round(thr['rate_per_sec']))
        m['ThroughputLatency'] = gfloat(thr['ms_per_pair'], 3)
    else:
        m['ThroughputSec']     = _MISSING
        m['ThroughputRate']    = _MISSING
        m['ThroughputLatency'] = _MISSING

    abl = d['ablation']
    # Scoped to the moderate/deployment-realistic τ₂ range (≤0.9) — the extended
    # near-1.0 τ₂ values exist only to drive the forced-escalation scenario
    # below (§5.1.2) and would make "even at the most permissive point" in the
    # §5.4.2 efficiency narrative misleading if mixed in here.
    m['TierThreeCallRateMax'] = gpct(abl[abl['tau2'] <= 0.9]['tier3_pct'].max(), 3)

    # ---- Error analysis ----
    total_errors = (vt['y_true'] != vt['y_pred']).sum()
    hard_errors  = ((vt['y_true'] != vt['y_pred']) & (vt['difficulty'] == 'hard')).sum()
    hard_err_pct = 100 * hard_errors / total_errors if total_errors else 0.0
    triv_err_pct = 100 - hard_err_pct

    m['CascadeTotalErrors']     = gint(total_errors)
    m['CascadeErrorsHardN']     = gint(hard_errors)
    m['CascadeErrorsTrivialN']  = gint(total_errors - hard_errors)
    m['CascadeErrorsHardPct']   = gpct(hard_err_pct)
    m['CascadeErrorsTrivialPct'] = gpct(triv_err_pct)

    # Hard subset performance
    hard_vt = vt[vt['difficulty'] == 'hard']
    if len(hard_vt):
        hard_acc = hard_vt['correct'].mean()
        hard_f1  = f1_score(hard_vt['y_true'], hard_vt['y_pred'], average='macro', zero_division=0)
        m['CascadeHardAcc'] = gfloat(hard_acc, 4)
        m['CascadeHardFone'] = gfloat(hard_f1,  4)
    else:
        m['CascadeHardAcc'] = _MISSING
        m['CascadeHardFone'] = _MISSING

    # ---- SVM error-by-difficulty (from svm_val_errors.csv) ----
    se = d['svm_errors']
    if se is not None:
        svm_total_errors = int((se['y_true'] != se['y_pred']).sum())
        svm_hard_errors  = int(((se['y_true'] != se['y_pred']) & (se['difficulty'] == 'hard')).sum())
        svm_hard_err_pct = 100 * svm_hard_errors / svm_total_errors if svm_total_errors else 0.0
        svm_triv_err_pct = 100 - svm_hard_err_pct

        m['SvmTotalErrors']    = gint(svm_total_errors)
        m['CascadeVsSvmRatio'] = str(round(svm_total_errors / total_errors)) if total_errors else _MISSING
        m['SvmErrorsHardN']    = gint(svm_hard_errors)
        m['SvmErrorsHardPct']  = gpct(svm_hard_err_pct)
        m['SvmErrorsTrivialN'] = gint(svm_total_errors - svm_hard_errors)
        m['SvmErrorsTrivialPct'] = gpct(svm_triv_err_pct)

        svm_textsim = se[se['difficulty'] == 'trivial_high_textsim']
        svm_hard    = se[se['difficulty'] == 'hard']
        if len(svm_textsim):
            m['SvmHighTextsimAcc']  = gfloat(svm_textsim['correct'].mean(), 4)
            m['SvmHighTextsimFone'] = gfloat(
                f1_score(svm_textsim['y_true'], svm_textsim['y_pred'], average='macro', zero_division=0), 4)
        else:
            m['SvmHighTextsimAcc'] = _MISSING
            m['SvmHighTextsimFone'] = _MISSING
        if len(svm_hard):
            m['SvmHardAcc']  = gfloat(svm_hard['correct'].mean(), 4)
            m['SvmHardFone'] = gfloat(
                f1_score(svm_hard['y_true'], svm_hard['y_pred'], average='macro', zero_division=0), 4)
        else:
            m['SvmHardAcc'] = _MISSING
            m['SvmHardFone'] = _MISSING
    else:
        for k in ('SvmTotalErrors', 'CascadeVsSvmRatio', 'SvmErrorsHardN', 'SvmErrorsHardPct',
                  'SvmErrorsTrivialN', 'SvmErrorsTrivialPct', 'SvmHighTextsimAcc', 'SvmHighTextsimFone',
                  'SvmHardAcc', 'SvmHardFone'):
            m[k] = _MISSING

    # ---- SVM baseline (from svm_val_metrics.json) ----
    if d['svm']:
        svm = d['svm']
        m['SvmFone']   = gfloat(svm['macro_f1'],      4)
        m['SvmAuc']    = gfloat(svm['auc_roc'],       4)
        m['SvmCsFone'] = gfloat(svm['cold_start_f1'], 4)
        lat = svm.get('latency_ms')
        m['SvmLatency'] = gfloat(lat / svm['n_val'], 5) if lat and svm.get('n_val') else _MISSING
    else:
        m['SvmFone']    = _MISSING
        m['SvmAuc']     = _MISSING
        m['SvmCsFone']  = _MISSING
        m['SvmLatency'] = _MISSING

    # ---- Node2Vec ablation (from node2vec_ablation.json) ----
    n2v = d['n2v_ablation']
    if n2v:
        without, with_n2v, coverage = n2v['without'], n2v['with'], n2v['coverage']
        m['NTwoVWithCallPct']   = gpct(with_n2v['tier1_call_rate_%'], 2)
        m['NTwoVWithCallN']     = gint(with_n2v['tier1_call_n'])
        m['NTwoVWithOverallFone'] = gfloat(with_n2v['macro_f1'], 4)
        m['NTwoVWithConfFone']    = gfloat(with_n2v['macro_f1_confident'], 4)
        m['NTwoVNoOverallFone']   = gfloat(without['macro_f1'], 4)
        m['NTwoVNoConfFone']      = gfloat(without['macro_f1_confident'], 4)
        m['NTwoVNoAccOverall']    = gfloat(without['accuracy'], 4)
        m['NTwoVWithAccOverall']  = gfloat(with_n2v['accuracy'], 4)
        m['NTwoVIdOneCoverage'] = gpct(coverage['id1_coverage_pct'], 1)
        m['NTwoVIdTwoCoverage'] = gpct(coverage['id2_coverage_pct'], 1)
        m['NTwoVZeroVecPct']    = gpct(coverage['zero_vec_pct'], 1)
        m['NTwoVZeroVecN']      = gint(coverage['zero_vec_n'])
    else:
        for k in ('NTwoVWithCallPct', 'NTwoVWithCallN', 'NTwoVWithOverallFone', 'NTwoVWithConfFone',
                  'NTwoVNoOverallFone', 'NTwoVNoConfFone', 'NTwoVNoAccOverall', 'NTwoVWithAccOverall',
                  'NTwoVIdOneCoverage', 'NTwoVIdTwoCoverage', 'NTwoVZeroVecPct', 'NTwoVZeroVecN'):
            m[k] = _MISSING

    n2v_tt = d['n2v_test_tiers']
    if n2v_tt is not None:
        m['NTwoVTestTierOnePct'] = gpct(100 * (n2v_tt['tier_used'] == 1).mean(), 1)
    else:
        m['NTwoVTestTierOnePct'] = _MISSING

    # ---- Ablation grid ----
    # Scoped to the original moderate τ₂ range (≤0.9) — §5.1's opening grid
    # description of "the grid"; the extended τ₂ values (0.95-0.9999) feed only
    # the forced-escalation scenario macros below and get their own count
    # (AblForceExtraPoints/AblForceGridPoints), so they must not silently widen
    # min/max/best-row selections meant to summarize the moderate grid.
    moderate_abl  = abl[abl['tau2'] <= 0.9]

    # τ₁∈{0.6,0.7} rows (low Tier-1 threshold → high call rate at Tier 1)
    low_t1 = moderate_abl[moderate_abl['tau1'].isin([0.6, 0.7])]
    # τ₁≥0.8, τ₂=0.5 (best F1 grid points)
    best_rows = moderate_abl[(moderate_abl['tau1'] >= 0.8) & (moderate_abl['tau2'] == 0.5)]
    default_row = moderate_abl[(moderate_abl['tau1'] == cascade_cfg['tier1_threshold']) & (moderate_abl['tau2'] == cascade_cfg['tier2_threshold'])].iloc[0]
    # τ₁∈{0.9,0.95} (higher break point)
    high_t1 = moderate_abl[moderate_abl['tau1'].isin([0.9, 0.95])]

    abl_grid_pts  = len(moderate_abl)
    m['AblGridPoints']         = str(abl_grid_pts)
    m['AblForceExtraPoints']   = str(len(abl) - abl_grid_pts)
    m['AblForceGridPoints']    = str(len(abl))
    m['AblTierOneLowFone']       = gfloat(low_t1['macro_f1'].min(), 4)
    m['AblTierOneHighFone']      = gfloat(low_t1['macro_f1'].max(), 4)
    m['AblTierOneLowPct']      = gpct(low_t1['tier1_pct'].iloc[0])
    m['AblTierOneBreakPct']    = gpct(moderate_abl[moderate_abl['tau1'] == cascade_cfg['tier1_threshold']]['tier1_pct'].iloc[0])
    m['AblTierOneBreakPctHigh'] = gpct(high_t1['tier1_pct'].iloc[0])
    m['AblBestFone']             = gfloat(best_rows['macro_f1'].max(), 4)
    m['AblDefaultFone']          = gfloat(default_row['macro_f1'], 4)
    # τ₁≥0.8 overall Macro F1 range (low end; high end coincides with AblBestFone)
    high_break_t1 = moderate_abl[moderate_abl['tau1'] >= 0.8]
    m['AblHighTauOneLowFone']        = gfloat(high_break_t1['macro_f1'].min(), 4)
    # max Tier-3 rate at τ₁=0.8 (across the moderate τ₂ sweep)
    t1_08 = moderate_abl[moderate_abl['tau1'] == cascade_cfg['tier1_threshold']]
    m['AblTauTwoMaxTierThreePct']  = gpct(t1_08['tier3_pct'].max(), 2)
    # Tier-3 call rates for the τ₂ table (τ₁=0.8 fixed); use letter names (no digits/underscores in TeX)
    _tau2_names = {0.5: 'ZeroFive', 0.6: 'ZeroSix', 0.7: 'ZeroSeven', 0.8: 'ZeroEight', 0.9: 'ZeroNine',
                   0.95: 'ZeroNineFive', 0.99: 'ZeroNineNine', 0.999: 'ZeroNineNineNine',
                   0.9999: 'ZeroNineNineNineNine'}
    for _, row in t1_08.sort_values('tau2').iterrows():
        tau2 = round(row['tau2'], 4)
        if tau2 not in _tau2_names:
            raise ValueError(f'No TeX-safe macro name mapped for tau2={tau2} — add one to _tau2_names')
        m[f'AblTThreeRate{_tau2_names[tau2]}'] = gpct(row['tier3_pct'], 3)

    # τ₁=0.8 row across the FULL extended grid — feeds the forced-escalation
    # scenario macros below (needs the near-1.0 τ₂ values that moderate_abl excludes).
    t1_08_full = abl[abl['tau1'] == cascade_cfg['tier1_threshold']]

    # ---- Forced-escalation scenario (thesis §5.1.1): push τ₂ towards 1.0 at the
    # deployed τ₁ to see how far the Tier-3 call rate can be driven, and what that
    # costs/buys in accuracy. Uses per-tier F1 columns added alongside the τ₂ grid
    # extension (0.9999 max) in ablate_cascade_thresholds.py.
    moderate_row = t1_08_full[np.isclose(t1_08_full['tau2'], 0.99)].iloc[0]
    plateau_row  = t1_08_full[np.isclose(t1_08_full['tau2'], 0.999)].iloc[0]

    m['AblForceModerateTierThreePct']  = gpct(moderate_row['tier3_pct'], 2)
    m['AblForceModerateFone']          = gfloat(moderate_row['macro_f1'], 4)
    m['AblForceModerateTierThreeFone'] = gfloat(moderate_row['tier3_f1'], 4)

    m['AblForcePlateauTierThreePct']  = gpct(plateau_row['tier3_pct'], 2)
    m['AblForcePlateauFone']          = gfloat(plateau_row['macro_f1'], 4)
    m['AblForcePlateauTierThreeFone'] = gfloat(plateau_row['tier3_f1'], 4)
    m['AblForcePlateauTierTwoPct']    = gpct(plateau_row['tier2_pct'], 2)

    m['AblForceFoneDrop'] = gfloat(default_row['macro_f1'] - plateau_row['macro_f1'], 4)
    # Tier-3's own accuracy at its smallest (default) vs. largest (forced) population
    m['AblForceTierThreeFoneGain'] = gfloat(plateau_row['tier3_f1'] - t3_f1, 4)

    # ---- Tier-2 confidence ceiling (explains the plateau above: a RandomForest
    # vote fraction is quantized to n_estimators steps, so a proba==1.0 pair
    # cannot be escalated by raising τ₂ no matter how close to 1.0 it gets) ----
    sat = d['tier2_saturation']
    if sat:
        m['TierTwoSatEligibleN']       = gint(sat['n_tier2_eligible'])
        m['TierTwoSatEligiblePct']     = gpct(100 * sat['n_tier2_eligible'] / n_val, 2)
        m['TierTwoSatCeilingPct']      = gpct(sat['pct_conf_eq_1.0'], 2)
        m['TierTwoSatNonSaturatedPct'] = gpct(100 - sat['pct_conf_eq_1.0'], 2)
        m['TierTwoSatNEstimators']     = str(sat['n_estimators'])
    else:
        for k in ('TierTwoSatEligibleN', 'TierTwoSatEligiblePct', 'TierTwoSatCeilingPct',
                  'TierTwoSatNonSaturatedPct', 'TierTwoSatNEstimators'):
            m[k] = _MISSING

    # ---- Standalone EmbeddingClassifier baseline (100%-to-Tier-3 reference point) ----
    if d['embedding']:
        emb = d['embedding']
        m['EmbeddingFone']   = gfloat(emb['macro_f1'], 4)
        m['EmbeddingAuc']    = gfloat(emb['auc_roc'], 4)
        m['EmbeddingCsFone'] = gfloat(emb['cold_start_f1'], 4)
    else:
        m['EmbeddingFone']   = _MISSING
        m['EmbeddingAuc']    = _MISSING
        m['EmbeddingCsFone'] = _MISSING

    # ---- Negative-sampling artifact audit: explains the near-perfect score above ----
    audit = d['neg_sampling_audit']
    if audit:
        ref = next(r for r in audit['by_threshold'] if r['threshold'] == audit['reference_threshold'])
        m['HubThreshold']   = gint(audit['reference_threshold'])
        m['HubCount']       = gint(ref['n_hub_id1'])
        m['HubUniqueSourceCount'] = gint(ref['n_unique_id1'])
        m['HubRowCount']    = gint(ref['hub_row_count'])
        m['HubRowPct']      = gfloat(ref['hub_row_pct'], 2)
        m['HubPurityPct']   = gfloat(100 * ref['hub_purity_fraction'], 2)
        m['NonHubRowCount'] = gint(ref['non_hub_row_count'])
        m['NonHubPosPct']   = gfloat(100 * ref['non_hub_positive_rate'], 2)
        m['HubLookupFone']  = gfloat(ref['trivial_lookup_macro_f1'], 6)
        m['HubLookupAcc']   = gfloat(ref['trivial_lookup_accuracy'], 6)
        m['HubTestOverlap'] = gint(audit['test_hub_id1_overlap'])
        m['HubTestRowPct']  = gfloat(audit['test_rows_with_hub_id1_pct'], 2)
    else:
        for k in ('HubThreshold', 'HubCount', 'HubUniqueSourceCount', 'HubRowCount', 'HubRowPct',
                  'HubPurityPct', 'NonHubRowCount', 'NonHubPosPct', 'HubLookupFone', 'HubLookupAcc',
                  'HubTestOverlap', 'HubTestRowPct'):
            m[k] = _MISSING

    hp = d['hub_in_predictions']
    if hp:
        m['HubValSourcedPct']       = gfloat(hp['val_hub_sourced_pct'], 2)
        m['HubValLookupFone']       = gfloat(hp['trivial_lookup_macro_f1_same_val_set'], 4)
        m['HubValAgreementPct']     = gfloat(hp['agreement_cascade_vs_trivial_lookup_pct'], 2)
        m['HubValHubAgreementPct']  = gfloat(hp['agreement_on_hub_subset_pct'], 2)
    else:
        for k in ('HubValSourcedPct', 'HubValLookupFone', 'HubValAgreementPct', 'HubValHubAgreementPct'):
            m[k] = _MISSING

    # ---- Fresh, artifact-free Wikipedia validation dataset ----
    wf = d['wikipedia']
    wf_name_to_macro = {'structural': 'WfStructural', 'tfidf': 'WfTfidf', 'pos': 'WfPos',
                        'embedding': 'WfEmbedding', 'svm': 'WfSvm', 'cascade': 'WfCascade'}
    if wf:
        for name, prefix in wf_name_to_macro.items():
            r = wf.get(name)
            m[f'{prefix}Fone'] = gfloat(r['macro_f1'], 4) if r else _MISSING
            m[f'{prefix}Auc']  = gfloat(r['auc_roc'], 4) if r and r.get('auc_roc') is not None else _MISSING
        cascade_tiers = wf.get('cascade', {}).get('tier_stats', {})
        for t, name in [('tier0', 'Zero'), ('tier1', 'One'), ('tier2', 'Two'), ('tier3', 'Three')]:
            stats = cascade_tiers.get(t)
            m[f'WfTier{name}Pct'] = gfloat(stats['pct'], 1) if stats else _MISSING
            m[f'WfTier{name}Count'] = gint(stats['n']) if stats else _MISSING
        meta = wf.get('_meta', {})
        m['WfTrainSplitSize'] = gint(meta.get('n_train', 0))
        m['WfValSplitSize']   = gint(meta.get('n_val', 0))
        m['WfGraphNodes']     = gint(meta.get('graph_nodes', 0))
        m['WfGraphEdges']     = gint(meta.get('graph_edges', 0))
        m['WfMeanDegree']     = gfloat(2 * meta.get('graph_edges', 0) / max(meta.get('graph_nodes', 1), 1), 1)
    else:
        for prefix in wf_name_to_macro.values():
            m[f'{prefix}Fone'] = m[f'{prefix}Auc'] = _MISSING
        for name in ('Zero', 'One', 'Two', 'Three'):
            m[f'WfTier{name}Pct'] = m[f'WfTier{name}Count'] = _MISSING
        for k in ('WfTrainSplitSize', 'WfValSplitSize', 'WfGraphNodes', 'WfGraphEdges', 'WfMeanDegree'):
            m[k] = _MISSING

    # ---- Fresh dataset: crawl + text-fetch provenance (dataset construction methodology) ----
    wc = d['wikipedia_crawl']
    if wc:
        m['WfCrawlSeed']            = str(wc['seed']).replace('_', ' ')
        m['WfCrawlTarget']          = gint(wc['target'])
        m['WfCrawlPageRows']        = gint(wc['n_page_rows_total'])
        m['WfCrawlPages']           = gint(wc['n_namespace0_nonredirect_pages'])
        m['WfCrawlLinktargetRows']  = gint(wc['n_linktarget_rows_total'])
        m['WfCrawlLinktargets']     = gint(wc['n_namespace0_link_targets'])
        m['WfCrawlPagelinksRows']   = gint(wc['n_pagelinks_rows_raw'])
        m['WfCrawlResolvedEdges']   = gint(wc['n_resolved_directed_edges'])
        m['WfCrawlPagesWithLink']   = gint(wc['n_pages_with_at_least_one_link'])
        m['WfCrawlNodes']           = gint(wc['n_crawled_nodes'])
        m['WfCrawlEdges']           = gint(wc['n_crawled_undirected_edges'])
        m['WfCrawlMeanDegree']      = gfloat(wc['crawled_mean_degree'], 1)
        sizes = wc['dump_sizes_bytes']
        m['WfDumpPageMb']       = gfloat(sizes['page_bytes'] / 1e6, 1)
        m['WfDumpLinktargetMb'] = gfloat(sizes['linktarget_bytes'] / 1e6, 1)
        m['WfDumpPagelinksMb']  = gfloat(sizes['pagelinks_bytes'] / 1e6, 1)
    else:
        for k in ('WfCrawlSeed', 'WfCrawlTarget', 'WfCrawlPageRows', 'WfCrawlPages',
                  'WfCrawlLinktargetRows', 'WfCrawlLinktargets', 'WfCrawlPagelinksRows',
                  'WfCrawlResolvedEdges', 'WfCrawlPagesWithLink', 'WfCrawlNodes', 'WfCrawlEdges',
                  'WfCrawlMeanDegree', 'WfDumpPageMb', 'WfDumpLinktargetMb', 'WfDumpPagelinksMb'):
            m[k] = _MISSING

    wt = d['wikipedia_text']
    if wt:
        m['WfTextConfig']    = str(wt['config'])
        m['WfTextWanted']    = gint(wt['n_titles_wanted'])
        m['WfTextMatched']   = gint(wt['n_titles_matched'])
        m['WfTextMatchPct']  = gfloat(wt['match_pct'], 1)
        m['WfTextEmptyN']    = gint(wt['n_nodes_empty_text'])
    else:
        for k in ('WfTextConfig', 'WfTextWanted', 'WfTextMatched', 'WfTextMatchPct', 'WfTextEmptyN'):
            m[k] = _MISSING

    # ---- Kaggle scores ----
    kg = d['kaggle']
    # Reference (structural-only) cascade — highest public score
    ref_rows = kg[~kg['description'].str.contains('Node2Vec', na=False)]
    n2v_rows = kg[kg['description'].str.contains('Node2Vec', na=False)]

    kaggle_our_private = None
    if not ref_rows.empty:
        best_ref = ref_rows.sort_values('public_score', ascending=False).iloc[0]
        m['KaggleBaselinePublic']  = gfloat(best_ref['public_score'], 5)
        m['KaggleBaselinePrivate'] = gfloat(best_ref['private_score'], 5)
        kaggle_our_private = float(best_ref['private_score'])
    else:
        m['KaggleBaselinePublic']  = _MISSING
        m['KaggleBaselinePrivate'] = _MISSING

    if not n2v_rows.empty:
        best_n2v = n2v_rows.sort_values('public_score', ascending=False).iloc[0]
        m['NTwoVKagglePublic']  = gfloat(best_n2v['public_score'], 5)
        m['NTwoVKagglePrivate'] = gfloat(best_n2v['private_score'], 5)
    else:
        m['NTwoVKagglePublic']  = _MISSING
        m['NTwoVKagglePrivate'] = _MISSING

    # ---- Shared raw intermediates for figure aggregation ----
    shared['G'] = G
    shared['y'] = y
    shared['not_self'] = not_self
    shared['cn_vals'] = cn_vals
    shared['tfidf_vals'] = tfidf_vals
    shared['diff_train_counts'] = diff_train_counts
    shared['n_train_pos'] = n_train_pos
    shared['n_train_neg'] = n_train_neg
    shared['kaggle_our_private'] = kaggle_our_private

    return m, shared


# ---------------------------------------------------------------------------
# Compute figures (raw numeric aggregates — no Greek formatting)
# ---------------------------------------------------------------------------

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

    # ---- node2vec ablation (already a small aggregate dict) ----
    figs['node2vec_ablation'] = d['n2v_ablation']

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

    return figs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    d = _load()
    m, shared = _compute_macros(d)
    figs = _compute_figures(d, shared)

    STATS.mkdir(parents=True, exist_ok=True)
    out = STATS / 'summary_stats.json'
    out.write_text(json.dumps({'macros': m, 'figures': figs}, indent=2))
    log.info('Wrote %d macros + %d figure groups → %s', len(m), len(figs), out)


if __name__ == '__main__':
    main()
