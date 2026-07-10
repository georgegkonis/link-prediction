"""
Generate paper/generated_macros.tex from experiment output CSVs.

Reads:
  data/raw/train.csv, test.csv                        — dataset sizes
  data/interim/difficulty_train.csv                   — difficulty breakdown (train)
  data/interim/difficulty_test.csv                    — difficulty breakdown (test)
  data/interim/difficulty_thresholds.json             — CN / TF-IDF thresholds
  data/interim/leakage_pairs.csv                      — leakage audit
  outputs/predictions/cascade_val_tiers.csv           — per-pair val predictions
  outputs/predictions/cascade_test_tiers.csv          — per-pair test predictions
  outputs/predictions/cascade_threshold_ablation.csv  — threshold sweep
  outputs/predictions/kaggle_scores.csv               — Kaggle leaderboard scores
  outputs/predictions/svm_val_metrics.json            — SVM scalar metrics

Writes:
  paper/generated_macros.tex

Usage:
    python -m scripts.generate_macros
"""

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Structural constants — cannot change without full pipeline re-run
# ---------------------------------------------------------------------------
_SVM_SAMPLE_SIZE  = 20_000
_N2V_DIM          = 68
_ABL_GRID_POINTS  = 25
_GRAPH_NODES_TOTAL = 837_855   # rows in nodes.tsv
_GRAPH_MEAN_DEGREE = 1.30       # measured via audit_leakage / graph build
_GRAPH_NODES_STRUCTURAL = 668_452   # nodes with ≥1 positive train edge


INTERIM     = pathlib.Path('data/interim')
RAW         = pathlib.Path('data/raw')
PREDICTIONS = pathlib.Path('outputs/predictions')
PAPER       = pathlib.Path('paper')

# ---------------------------------------------------------------------------
# Greek number formatters
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


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load() -> dict:
    print('Loading CSVs...')
    d: dict = {}

    d['train'] = pd.read_csv(RAW / 'train.csv')
    d['test']  = pd.read_csv(RAW / 'test.csv')

    d['diff_train'] = pd.read_csv(INTERIM / 'difficulty_train.csv', index_col='id')
    d['diff_test']  = pd.read_csv(INTERIM / 'difficulty_test.csv',  index_col='id')

    thresh_path = INTERIM / 'difficulty_thresholds.json'
    if thresh_path.exists():
        d['thresholds'] = json.loads(thresh_path.read_text())
    else:
        print('  WARNING: difficulty_thresholds.json not found — run make analyze-dataset first')
        d['thresholds'] = {'cn_threshold': 0.0, 'tfidf_threshold': 0.098}

    d['leakage']  = pd.read_csv(INTERIM / 'leakage_pairs.csv')
    d['val_tiers'] = pd.read_csv(PREDICTIONS / 'cascade_val_tiers.csv')
    d['test_tiers'] = pd.read_csv(PREDICTIONS / 'cascade_test_tiers.csv')
    d['ablation'] = pd.read_csv(PREDICTIONS / 'cascade_threshold_ablation.csv')
    d['kaggle']   = pd.read_csv(PREDICTIONS / 'kaggle_scores.csv')

    svm_path = PREDICTIONS / 'svm_val_metrics.json'
    if svm_path.exists():
        d['svm'] = json.loads(svm_path.read_text())
    else:
        print('  WARNING: svm_val_metrics.json not found — run make train MODEL=svm first')
        d['svm'] = None

    return d


# ---------------------------------------------------------------------------
# Compute macros
# ---------------------------------------------------------------------------

def _compute(d: dict) -> dict[str, str]:
    m: dict[str, str] = {}

    # ---- Dataset sizes ----
    n_train_raw   = len(d['train'])
    n_test        = len(d['test'])
    self_loop_mask = d['train']['id1'] == d['train']['id2']
    n_self_loops  = self_loop_mask.sum()
    n_no_self     = n_train_raw - n_self_loops

    y = d['train']['label'].values
    not_self_idx = np.where(~self_loop_mask.values)[0]
    tr, val = train_test_split(not_self_idx, test_size=0.2, stratify=y[not_self_idx], random_state=42)
    n_tr, n_val = len(tr), len(val)

    m['TrainPairs']       = gint(n_train_raw)
    m['TrainPairsNoSelf'] = gint(n_no_self)
    m['TrainSplitSize']   = gint(n_tr)
    m['ValSplitSize']     = gint(n_val)
    m['TestPairs']        = gint(n_test)
    m['SvmSampleSize']    = gint(_SVM_SAMPLE_SIZE)
    m['GraphNodes']       = gint(_GRAPH_NODES_STRUCTURAL)
    m['GraphMeanDegree']  = gfloat(_GRAPH_MEAN_DEGREE, 2)
    m['GraphNodesTotal']  = gint(_GRAPH_NODES_TOTAL)

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
    m['LeakageExactPct']       = gfloat(100 * exact / n_test, 4)
    m['LeakageReversed']       = gint(reversed_)
    m['LeakageReversedPct']    = gfloat(100 * reversed_ / n_test, 4)
    m['SelfLoopsTrain']        = gint(n_self_loops)
    m['SelfLoopsTrainPos']     = gint(self_pos)
    m['SelfLoopsTrainNeg']     = str(int(self_neg))
    m['SelfLoopsTest']         = gint(test_self)
    m['IntraTrainDuplicates']  = gint(n_dup_groups)

    # ---- Separability thresholds ----
    cn_thr    = d['thresholds']['cn_threshold']
    tfidf_thr = d['thresholds']['tfidf_threshold']
    m['CnThreshold']    = gfloat(cn_thr, 3)
    m['TfidfThreshold'] = gfloat(tfidf_thr, 3)

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

    # ---- Cascade default thresholds ----
    m['TauOneDefault'] = '0{,}8'
    m['TauTwoDefault'] = '0{,}7'
    m['NTwoVDim']      = str(_N2V_DIM)

    # ---- Cascade validation results ----
    vt = d['val_tiers']
    n_vt = len(vt)

    def _tier_stats(tier_id):
        sub = vt[vt['tier_used'] == tier_id]
        n   = len(sub)
        if n == 0:
            return n, 0.0, float('nan'), float('nan')
        acc  = sub['correct'].mean()
        from sklearn.metrics import f1_score
        f1   = f1_score(sub['y_true'], sub['y_pred'], average='macro', zero_division=0)
        return n, 100 * n / n_vt, acc, f1

    t1_n, t1_pct, t1_acc, t1_f1 = _tier_stats(1)
    t2_n, t2_pct, t2_acc, t2_f1 = _tier_stats(2)
    t3_n, t3_pct, t3_acc, t3_f1 = _tier_stats(3)

    from sklearn.metrics import f1_score, roc_auc_score
    cascade_f1  = f1_score(vt['y_true'], vt['y_pred'], average='macro', zero_division=0)
    cascade_auc = roc_auc_score(vt['y_true'], vt['y_pred'])

    m['CascadeValFone']  = gfloat(cascade_f1,  4)
    m['CascadeValAuc'] = gfloat(cascade_auc, 4)

    m['TierOneCount'] = gint(t1_n)
    m['TierOnePct']   = gpct(t1_pct)
    m['TierOneAcc']   = gfloat(t1_acc, 4)
    m['TierOneFone']    = gfloat(t1_f1,  4)

    m['TierTwoCount'] = gint(t2_n)
    m['TierTwoPct']   = gpct(t2_pct)
    m['TierTwoAcc']   = gfloat(t2_acc, 4)
    m['TierTwoFone']    = gfloat(t2_f1,  4)

    m['TierThreeCount'] = gint(t3_n)
    m['TierThreePct']   = gpct(t3_pct, 2)
    m['TierThreeAcc']   = gfloat(t3_acc, 4)
    m['TierThreeFone']    = gfloat(t3_f1,  4)

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

    # ---- Tier × difficulty cross-tab ----
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
                m[f'Tier{tier_name}{diff_name}Acc'] = '---'

    # ---- Cold-start (no common neighbours) ----
    # proxy: Tier 1 passes only pairs with structural signal, Tier 2/3 are cold-start
    # actual cold-start = pairs with cn == 0 in val set; we read from val_tiers difficulty
    # (trivial_high_cn pairs DO have structural signal; hard pairs might or might not)
    # Simplest proxy matching the paper: pairs routed to Tier 2 or 3 are effectively cold-start
    # The paper states 99.9963% — compute directly
    cs_count  = (vt['tier_used'] >= 2).sum()
    cs_pct    = 100 * cs_count / n_vt
    non_cs    = n_vt - cs_count
    m['ColdStartCount']    = gint(cs_count)
    m['ColdStartPct']      = gfloat(cs_pct, 4)
    m['ColdStartNonCount'] = gint(non_cs)

    # ---- Hard residual (Tier 3 ∩ hard) ----
    hard_res = vt[(vt['tier_used'] == 3) & (vt['difficulty'] == 'hard')]
    n_hr     = len(hard_res)
    if n_hr > 0:
        hr_acc = hard_res['correct'].mean()
    else:
        hr_acc = float('nan')

    m['HardResCount'] = gint(n_hr)
    m['HardResAcc']   = gfloat(hr_acc, 4) if n_hr > 0 else '---'

    # These sub-metrics come from manual analysis in the paper; keep as constants
    # (they'd require merging with nodes.tsv which is 641MB)
    m['HardResMissingN']      = '37'
    m['HardResMissingPct']    = '47{,}4'
    m['HardResCompleteN']     = '41'
    m['HardResCompleteAcc']   = '0{,}0732'
    m['HardResIncompleteAcc'] = '0{,}2703'
    m['HardResNodeId']        = '415071'
    m['HardResNodePairs']     = '18'
    m['HardResNodePredPosPct'] = '55{,}6'

    # ---- Test-set tier distribution ----
    tt = d['test_tiers']
    n_tt = len(tt)
    tt1_n   = (tt['tier_used'] == 1).sum()
    tt0_n   = (tt['tier_used'] == 0).sum()
    m['TestTierOneCount'] = gint(tt1_n)
    m['TestTierOnePct']   = gpct(100 * tt1_n / n_tt, 1)
    m['TestTierZeroPct']  = gpct(100 * tt0_n / n_tt, 1)

    # ---- Efficiency / throughput (constants from timing run) ----
    m['ThroughputSec']        = '1{,}68'
    m['ThroughputRate']       = '111.600'
    m['ThroughputLatency']    = '0{,}009'
    m['TierThreeCallRateMax'] = '0{,}674'

    # ---- Error analysis ----
    total_errors = (vt['y_true'] != vt['y_pred']).sum()
    hard_errors  = ((vt['y_true'] != vt['y_pred']) & (vt['difficulty'] == 'hard')).sum()
    triv_errors  = total_errors - hard_errors
    hard_err_pct = 100 * hard_errors / total_errors if total_errors else 0.0
    triv_err_pct = 100 - hard_err_pct

    m['CascadeTotalErrors']     = gint(total_errors)
    m['CascadeErrorsHardN']     = gint(hard_errors)
    m['CascadeErrorsHardPct']   = gpct(hard_err_pct)
    m['CascadeErrorsTrivialPct'] = gpct(triv_err_pct)

    # Hard subset performance
    hard_vt = vt[vt['difficulty'] == 'hard']
    if len(hard_vt):
        hard_acc = hard_vt['correct'].mean()
        hard_f1  = f1_score(hard_vt['y_true'], hard_vt['y_pred'], average='macro', zero_division=0)
        m['CascadeHardAcc'] = gfloat(hard_acc, 4)
        m['CascadeHardFone']  = gfloat(hard_f1,  4)
    else:
        m['CascadeHardAcc'] = '---'
        m['CascadeHardFone']  = '---'

    # SVM error stats (constants from paper — require separate SVM val run)
    m['SvmTotalErrors']     = gint(16_696)
    m['CascadeVsSvmRatio']  = '64'
    m['SvmErrorsHardPct']   = gpct(93.63)
    m['SvmErrorsTrivialPct'] = gpct(6.37)
    m['SvmHighTextsimAcc']  = gfloat(0.9838, 4)
    m['SvmHighTextsimFone']   = gfloat(0.4959, 4)
    m['SvmHardAcc']         = gfloat(0.8717, 4)
    m['SvmHardFone']          = gfloat(0.7035, 4)

    # ---- SVM baseline (from svm_val_metrics.json) ----
    if d['svm']:
        svm = d['svm']
        m['SvmFone']      = gfloat(svm['macro_f1'],      4)
        m['SvmAuc']     = gfloat(svm['auc_roc'],       4)
        m['SvmCsFone']    = gfloat(svm['cold_start_f1'], 4)
        lat = svm.get('latency_ms')
        m['SvmLatency'] = gfloat(lat / svm['n_val'], 5) if lat and svm.get('n_val') else '---'
    else:
        m['SvmFone']      = '---'
        m['SvmAuc']     = '---'
        m['SvmCsFone']    = '---'
        m['SvmLatency'] = '---'

    # ---- Node2Vec ablation (constants — come from train.py run with n2v) ----
    m['NTwoVWithCallPct']   = '99{,}30'
    m['NTwoVWithCallN']     = gint(186_298)
    m['NTwoVWithOverallFone'] = '0{,}9958'
    m['NTwoVWithConfFone']    = '0{,}9970'
    m['NTwoVNoOverallFone']   = '0{,}7693'
    m['NTwoVNoConfFone']      = '0{,}4996'
    m['NTwoVIdOneCoverage'] = '9{,}4'
    m['NTwoVIdTwoCoverage'] = '30{,}3'
    m['NTwoVZeroVecPct']    = '98{,}3'
    m['NTwoVZeroVecN']      = gint(234_276)
    m['NTwoVTestTierOnePct'] = '98{,}6'

    # ---- Ablation grid ----
    abl = d['ablation']
    # τ₁∈{0.6,0.7} rows (low Tier-1 threshold → high call rate at Tier 1)
    low_t1 = abl[abl['tau1'].isin([0.6, 0.7])]
    # τ₁≥0.8, τ₂=0.5 (best F1 grid points)
    best_rows = abl[(abl['tau1'] >= 0.8) & (abl['tau2'] == 0.5)]
    default_row = abl[(abl['tau1'] == 0.8) & (abl['tau2'] == 0.7)].iloc[0]
    # τ₁∈{0.9,0.95} (higher break point)
    high_t1 = abl[abl['tau1'].isin([0.9, 0.95])]

    abl_grid_pts  = len(abl)
    m['AblGridPoints']         = str(abl_grid_pts)
    m['AblTierOneLowFone']       = gfloat(low_t1['macro_f1'].min(), 4)
    m['AblTierOneHighFone']      = gfloat(low_t1['macro_f1'].max(), 4)
    m['AblTierOneLowPct']      = gpct(low_t1['tier1_pct'].iloc[0])
    m['AblTierOneBreakPct']    = gpct(abl[abl['tau1'] == 0.8]['tier1_pct'].iloc[0])
    m['AblTierOneBreakPctHigh'] = gpct(high_t1['tier1_pct'].iloc[0])
    m['AblBestFone']             = gfloat(best_rows['macro_f1'].max(), 4)
    m['AblDefaultFone']          = gfloat(default_row['macro_f1'], 4)
    # max Tier-3 rate at τ₁=0.8 (across τ₂ sweep)
    t1_08 = abl[abl['tau1'] == 0.8]
    m['AblTauTwoMaxTierThreePct']  = gpct(t1_08['tier3_pct'].max(), 2)
    # Tier-3 call rates for the τ₂ table (τ₁=0.8 fixed); use letter names (no digits/underscores in TeX)
    _tau2_names = {0.5: 'ZeroFive', 0.6: 'ZeroSix', 0.7: 'ZeroSeven', 0.8: 'ZeroEight', 0.9: 'ZeroNine'}
    for _, row in t1_08.sort_values('tau2').iterrows():
        letter = _tau2_names.get(round(row['tau2'], 1), str(row['tau2']).replace('.', 'p'))
        m[f'AblTThreeRate{letter}'] = gpct(row['tier3_pct'], 3)

    # ---- Kaggle scores ----
    kg = d['kaggle']
    # Reference (structural-only) cascade — highest public score
    ref_rows = kg[~kg['description'].str.contains('Node2Vec', na=False)]
    n2v_rows = kg[kg['description'].str.contains('Node2Vec', na=False)]

    if not ref_rows.empty:
        best_ref = ref_rows.sort_values('public_score', ascending=False).iloc[0]
        m['KaggleBaselinePublic']  = gfloat(best_ref['public_score'], 5)
        m['KaggleBaselinePrivate'] = gfloat(best_ref['private_score'], 5)
    else:
        m['KaggleBaselinePublic']  = '---'
        m['KaggleBaselinePrivate'] = '---'

    if not n2v_rows.empty:
        best_n2v = n2v_rows.sort_values('public_score', ascending=False).iloc[0]
        m['NTwoVKagglePublic']  = gfloat(best_n2v['public_score'], 5)
        m['NTwoVKagglePrivate'] = gfloat(best_n2v['private_score'], 5)
    else:
        m['NTwoVKagglePublic']  = '---'
        m['NTwoVKagglePrivate'] = '---'

    return m


# ---------------------------------------------------------------------------
# Emit TeX
# ---------------------------------------------------------------------------

_GROUPS = [
    ('Dataset sizes',          ['TrainPairs', 'TrainPairsNoSelf', 'TrainSplitSize', 'ValSplitSize',
                                 'TestPairs', 'SvmSampleSize', 'GraphNodes', 'GraphMeanDegree',
                                 'GraphNodesTotal']),
    ('Leakage audit',          ['LeakageExact', 'LeakageExactPct', 'LeakageReversed',
                                 'LeakageReversedPct', 'SelfLoopsTrain', 'SelfLoopsTrainPos',
                                 'SelfLoopsTrainNeg', 'SelfLoopsTest', 'IntraTrainDuplicates']),
    ('Separability thresholds',['CnThreshold', 'TfidfThreshold']),
    ('Difficulty — train',     ['TrainDiffSelfLoopN', 'TrainDiffSelfLoopPct', 'TrainDiffHighCnN',
                                 'TrainDiffHighCnPct', 'TrainDiffHighTextsimN', 'TrainDiffHighTextsimPct',
                                 'TrainDiffHardN', 'TrainDiffHardPct', 'TrainDiffTrivialPct']),
    ('Difficulty — test',      ['TestDiffSelfLoopN', 'TestDiffSelfLoopPct', 'TestDiffHighCnN',
                                 'TestDiffHighTextsimN', 'TestDiffHighTextsimPct',
                                 'TestDiffHardN', 'TestDiffHardPct']),
    ('Cascade thresholds',     ['TauOneDefault', 'TauTwoDefault', 'NTwoVDim']),
    ('Cascade val results',    ['CascadeValFone', 'CascadeValAuc',
                                 'TierOneCount', 'TierOnePct', 'TierOneAcc', 'TierOneFone',
                                 'TierTwoCount', 'TierTwoPct', 'TierTwoAcc', 'TierTwoFone',
                                 'TierThreeCount', 'TierThreePct', 'TierThreeAcc', 'TierThreeFone',
                                 'TierOnePosPct', 'TierThreeTruePosPct', 'TierThreePredPosPct']),
    ('Tier x difficulty',      ['TierOneHardN', 'TierOneHardAcc', 'TierOneHighTextsimN', 'TierOneHighTextsimAcc',
                                 'TierOneHighCnN', 'TierOneHighCnAcc',
                                 'TierTwoHardN', 'TierTwoHardAcc', 'TierTwoHighTextsimN', 'TierTwoHighTextsimAcc',
                                 'TierThreeHardN', 'TierThreeHardAcc',
                                 'TierThreeHighTextsimN', 'TierThreeHighTextsimAcc']),
    ('Test-set tier dist',     ['TestTierOneCount', 'TestTierOnePct', 'TestTierZeroPct']),
    ('Cold-start',             ['ColdStartCount', 'ColdStartPct', 'ColdStartNonCount']),
    ('Hard residual',          ['HardResCount', 'HardResAcc', 'HardResMissingN', 'HardResMissingPct',
                                 'HardResCompleteN', 'HardResCompleteAcc', 'HardResIncompleteAcc',
                                 'HardResNodeId', 'HardResNodePairs', 'HardResNodePredPosPct']),
    ('Efficiency',             ['ThroughputSec', 'ThroughputRate', 'ThroughputLatency', 'TierThreeCallRateMax']),
    ('Error analysis',         ['CascadeTotalErrors', 'CascadeErrorsHardN', 'CascadeErrorsHardPct',
                                 'CascadeErrorsTrivialPct', 'CascadeHardAcc', 'CascadeHardFone',
                                 'SvmTotalErrors', 'CascadeVsSvmRatio', 'SvmErrorsHardPct',
                                 'SvmErrorsTrivialPct', 'SvmHighTextsimAcc', 'SvmHighTextsimFone',
                                 'SvmHardAcc', 'SvmHardFone']),
    ('Node2Vec ablation',      ['NTwoVWithCallPct', 'NTwoVWithCallN', 'NTwoVWithOverallFone',
                                 'NTwoVWithConfFone', 'NTwoVNoOverallFone', 'NTwoVNoConfFone',
                                 'NTwoVIdOneCoverage', 'NTwoVIdTwoCoverage', 'NTwoVZeroVecPct',
                                 'NTwoVZeroVecN', 'NTwoVTestTierOnePct']),
    ('Threshold ablation',     ['AblGridPoints', 'AblTierOneLowFone', 'AblTierOneHighFone',
                                 'AblTierOneLowPct', 'AblTierOneBreakPct', 'AblTierOneBreakPctHigh',
                                 'AblBestFone', 'AblDefaultFone', 'AblTauTwoMaxTierThreePct',
                                 'AblTThreeRateZeroFive', 'AblTThreeRateZeroSix', 'AblTThreeRateZeroSeven',
                                 'AblTThreeRateZeroEight', 'AblTThreeRateZeroNine']),
    ('SVM baseline',           ['SvmFone', 'SvmAuc', 'SvmCsFone', 'SvmLatency']),
    ('Kaggle scores',          ['KaggleBaselinePublic', 'KaggleBaselinePrivate',
                                 'NTwoVKagglePublic', 'NTwoVKagglePrivate']),
]


def _emit_tex(macros: dict[str, str]) -> None:
    lines = [
        '% AUTO-GENERATED by scripts/generate_macros.py — do not edit by hand.',
        '% Re-run: make generate-macros',
        '',
    ]
    emitted = set()
    for group_name, keys in _GROUPS:
        lines.append(f'%% {group_name}')
        for key in keys:
            if key in macros:
                lines.append(f'\\newcommand{{\\{key}}}{{{macros[key]}}}')
                emitted.add(key)
        lines.append('')

    # Catch any computed keys not in the explicit group list
    remaining = {k: v for k, v in macros.items() if k not in emitted}
    if remaining:
        lines.append('%% Additional computed macros')
        for k, v in sorted(remaining.items()):
            lines.append(f'\\newcommand{{\\{k}}}{{{v}}}')
        lines.append('')

    out = PAPER / 'generated_macros.tex'
    out.write_text('\n'.join(lines))
    print(f'Wrote {len(macros)} macros → {out}')


def main():
    d = _load()
    macros = _compute(d)
    _emit_tex(macros)


if __name__ == '__main__':
    main()
