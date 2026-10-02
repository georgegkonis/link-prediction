"""
Compute statistics, metrics, and aggregates for figures and macros.

Reads:
    data/raw/dsaa/train.csv
    data/raw/dsaa/test.csv
    data/raw/dsaa/nodes.tsv
    data/interim/dsaa/difficulty_train.csv
    data/interim/dsaa/difficulty_test.csv
    data/interim/dsaa/difficulty_thresholds.json
    data/interim/dsaa/leakage_pairs.csv
    data/interim/dsaa/structural_train.csv
    data/interim/dsaa/tfidf_train.csv
    outputs/predictions/dsaa/cascade_val_tiers.csv
    outputs/predictions/dsaa/cascade_test_tiers.csv
    outputs/predictions/dsaa/cascade_threshold_ablation.csv
    outputs/predictions/dsaa/cascade_val_metrics.json
    outputs/predictions/dsaa/kaggle_scores.csv
    outputs/predictions/dsaa/svm_val_metrics.json
    outputs/predictions/dsaa/svm_val_errors.csv
    outputs/predictions/dsaa/hard_residual_analysis.json
    outputs/predictions/dsaa/throughput_benchmark.json
    outputs/predictions/dsaa/tier2_confidence_saturation.json
    outputs/predictions/dsaa/embedding_val_metrics.json
    outputs/predictions/graph_holdout_v1/matched_samples_v2/metrics.csv
    outputs/stats/{negative_sampling_audit,hub_in_predictions_audit,supervisor_audit}.json
    outputs/stats/wiki_cs_8k_experiment_results.json
    outputs/stats/wiki_cs_8k_sparse20{,_mixed}_results.json
    outputs/predictions/wiki_cs_8k_sparse20_mixed/test_hard_predictions.csv
    data/raw/wiki_cs_8k/{crawl_stats,text_fetch_stats}.json
    configs/{config,features,training,model} YAML defaults for legacy results
    DSAA feature/model run configuration snapshots when available
Writes:
    latex/shared/results/summary_stats.json

Usage:
    python -m scripts.thesis.compute_summary_stats
"""

import json
import pathlib

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.thesis._figure_data import DIFF_ORDER, _MODEL_NAMES, _compute_figures
from src.data.loader import build_graph
from src.utils.log_utils import setup_logging

log = setup_logging('compute_summary_stats')

INTERIM     = pathlib.Path('data/interim/dsaa')
RAW         = pathlib.Path('data/raw/dsaa')
PREDICTIONS = pathlib.Path('outputs/predictions/dsaa')
MATCHED_PREDICTIONS = pathlib.Path(
    'outputs/predictions/graph_holdout_v1/matched_samples_v2')
STATS       = pathlib.Path('outputs/stats')
CONFIGS     = pathlib.Path('configs')
PUBLISHED_RESULTS = pathlib.Path('latex/shared/results')
SPARSE_RANDOM = 'wiki_cs_8k_sparse20'
SPARSE_MIXED = 'wiki_cs_8k_sparse20_mixed'
SPARSE_BOOTSTRAP_RESAMPLES = 500


def _load_configs() -> dict:
    """Use recorded run settings; support existing results without snapshots."""
    cfg = yaml.safe_load((CONFIGS / 'config.yaml').read_text())
    cfg['training'] = yaml.safe_load((CONFIGS / 'training/default.yaml').read_text())
    cfg['features'] = yaml.safe_load((CONFIGS / 'features/default.yaml').read_text())
    cfg['model'] = {
        name: yaml.safe_load((CONFIGS / f'model/{name}.yaml').read_text())
        for name in _MODEL_NAMES
    }
    paths = {
        name: PREDICTIONS / f'{name}_run_config.json' for name in _MODEL_NAMES
    }
    paths['structural_features'] = INTERIM / 'structural_config.json'
    paths['semantic_features'] = INTERIM / 'semantic_config.json'
    present = {name for name, path in paths.items() if path.exists()}
    if not present:
        log.warning('No run configuration snapshots found; legacy results use YAML defaults')
        return cfg
    if present != set(paths):
        missing = sorted(set(paths) - present)
        raise ValueError(f'Incomplete run configuration snapshots: {missing}. '
                         'Regenerate all DSAA features and models before building the thesis.')

    runs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    reference = runs['cascade']['config']
    for name in _MODEL_NAMES:
        run = runs[name]
        actual = run['config']
        if actual['model']['name'] != name:
            raise ValueError(f'Run configuration does not match model {name}')
        for key in ('seed', 'training'):
            if actual[key] != reference[key]:
                raise ValueError(f'Inconsistent {key} across DSAA model runs')
        if pathlib.Path(actual['paths']['predictions']).resolve() != PREDICTIONS.resolve():
            raise ValueError(f'Model {name} used a different prediction directory')
        if pathlib.Path(run['raw_path']).resolve() != RAW.resolve() or (
                pathlib.Path(run['interim_path']).resolve() != INTERIM.resolve()):
            raise ValueError(f'Model {name} used different data paths from this thesis build')
        cfg['model'][name] = actual['model']

    for name in ('structural_features', 'semantic_features'):
        actual = runs[name]['config']
        if any((actual['dev']['nrows'] is not None,
                actual['dev']['skip_st'], actual['dev']['skip_pos'])):
            raise ValueError(f'{name} was produced by a development feature run')
        if pathlib.Path(actual['paths']['raw']).resolve() != RAW.resolve() or (
                pathlib.Path(actual['paths']['interim']).resolve() != INTERIM.resolve()):
            raise ValueError(f'{name} used different data paths from this thesis build')
    cfg['seed'] = reference['seed']
    cfg['training'] = reference['training']
    cfg['features'] = runs['semantic_features']['config']['features']
    return cfg

_MISSING = '---'

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
    sign = '-' if s.startswith('-') else ''
    i, f = s.lstrip('-').split('.')
    i_fmt = f'{int(i):,}'.replace(',', '.')
    return f'{sign}{i_fmt}{{,}}{f}'


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


def _sparse_results() -> tuple[dict, dict]:
    """Load comparable sparse Wiki runs and verify their fixed test suites."""
    runs = []
    for name in (SPARSE_RANDOM, SPARSE_MIXED):
        path = STATS / f'{name}_results.json'
        if not path.exists():
            raise FileNotFoundError(f'{path} not found — run make pipeline-wiki-sparse'
                                    ' and make pipeline-wiki-sparse-mixed first')
        runs.append(json.loads(path.read_text()))
    random_run, mixed_run = runs
    a, b = random_run['benchmark'], mixed_run['benchmark']
    if a['protocol'] != b['protocol'] or a['protocol'] != 'wiki_sparse_holdout_v1':
        raise ValueError('Sparse Wiki runs have different protocols')
    for key in ('seed', 'edge_retention', 'source'):
        if a[key] != b[key]:
            raise ValueError(f'Sparse Wiki runs differ on {key}')
    for key in ('observed_edges', 'test_random', 'test_hard'):
        if a['files'][key]['sha256'] != b['files'][key]['sha256']:
            raise ValueError(f'Sparse Wiki runs have different {key} files')
    for key in ('observed_positive_edges', 'held_out_positive_edges',
                'train_pairs', 'test_random_pairs', 'test_hard_pairs'):
        if a['counts'][key] != b['counts'][key]:
            raise ValueError(f'Sparse Wiki runs differ on {key}')
    if (a['negative_sampling']['train'] != 'uniform_verified_nonedge' or
            b['negative_sampling']['train'] !=
            'half_uniform_half_observed_two_hop_verified_nonedge'):
        raise ValueError('Sparse Wiki training-negative protocols are unexpected')
    if random_run['_meta']['run_config'] != mixed_run['_meta']['run_config']:
        raise ValueError('Sparse Wiki runs used different model or feature settings')
    return random_run, mixed_run


def _paired_sparse_hard_delta(
    mixed_run: dict, n_boot: int = SPARSE_BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float]:
    """Compare CascadeLP with embeddings on the same hard-test rows."""
    path = pathlib.Path('outputs/predictions') / SPARSE_MIXED / 'test_hard_predictions.csv'
    predictions = pd.read_csv(path, usecols=['y_true', 'cascade_pred', 'embedding_pred'])
    y = predictions['y_true'].to_numpy(dtype=np.int8)
    cascade = predictions['cascade_pred'].to_numpy(dtype=np.int8)
    embedding = predictions['embedding_pred'].to_numpy(dtype=np.int8)
    expected = mixed_run['suites']['test_hard']
    if (len(y) != expected['cascade']['n'] or
            not set(np.unique(np.concatenate((y, cascade, embedding)))) <= {0, 1}):
        raise ValueError('Sparse hard-test predictions have unexpected size or labels')
    cascade_f1 = f1_score(y, cascade, average='macro')
    embedding_f1 = f1_score(y, embedding, average='macro')
    if (not np.isclose(cascade_f1, expected['cascade']['macro_f1']) or
            not np.isclose(embedding_f1, expected['embedding']['macro_f1'])):
        raise ValueError('Sparse hard-test predictions disagree with the result summary')

    codes = [2 * y + model for model in (cascade, embedding)]
    rng = np.random.default_rng(mixed_run['benchmark']['seed'])
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        scores = []
        for code in codes:
            tn, fp, fn, tp = np.bincount(code[idx], minlength=4)
            scores.append(0.5 * (2 * tn / (2 * tn + fp + fn) +
                                 2 * tp / (2 * tp + fp + fn)))
        deltas[i] = scores[0] - scores[1]
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return cascade_f1 - embedding_f1, float(lo), float(hi)


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
        log.warning('difficulty_thresholds.json not found — run make run SCRIPT=dsaa.label_difficulty first')
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
    d['hard_residual'] = _load_json_optional(
        PREDICTIONS / 'hard_residual_analysis.json',
        'run python -m scripts.dsaa.analyze_hard_residual first')
    d['throughput'] = _load_json_optional(
        PREDICTIONS / 'throughput_benchmark.json',
        'run python -m scripts.dsaa.benchmark_inference first')
    d['tier2_saturation'] = _load_json_optional(
        PREDICTIONS / 'tier2_confidence_saturation.json',
        'run python -m scripts.dsaa.sweep_routing_thresholds first')
    d['embedding'] = _load_json_optional(
        PREDICTIONS / 'embedding_val_metrics.json', 'run make train MODEL=embedding first')
    d['cascade_val_metrics'] = _load_json_optional(
        PREDICTIONS / 'cascade_val_metrics.json', 'run make train MODEL=cascade first')
    d['baseline_metrics'] = {
        name: _load_json_optional(
            PREDICTIONS / f'{name}_val_metrics.json',
            f'run make train MODEL={name} first')
        for name in _MODEL_NAMES
    }
    d['matched_samples'] = _load_csv_optional(
        MATCHED_PREDICTIONS / 'metrics.csv',
        'run make compare-matched first')

    # ---- Negative-sampling artifact audit (explains the near-perfect DSAA 2023 score) ----
    d['neg_sampling_audit'] = _load_json_optional(
        STATS / 'negative_sampling_audit.json', 'run python -m scripts.dsaa.audit_negative_sampling first')
    d['hub_in_predictions'] = _load_json_optional(
        STATS / 'hub_in_predictions_audit.json', 'run python -m scripts.dsaa.audit_prediction_shortcut first')

    # ---- Cached-data protocol audit: graph-construction leakage + endpoint-order
    # sensitivity of the legacy DSAA 2023 row split (Chapter 4/5 limitations) ----
    d['protocol_audit'] = _load_json_optional(
        STATS / 'supervisor_audit.json',
        'run python -m scripts.dsaa.audit_protocol --swap first')

    # ---- Fresh, artifact-free Wikipedia validation dataset ----
    d['wikipedia'] = _load_json_optional(
        STATS / 'wiki_cs_8k_experiment_results.json',
        'run python -m scripts.wiki.run_experiment first')
    d['wiki_sparse_random'], d['wiki_sparse_mixed'] = _sparse_results()
    if d['wikipedia']:
        wiki_cfg = d['wikipedia'].get('_meta', {}).get('run_config')
        if wiki_cfg:
            if (wiki_cfg['features'] != d['cfg']['features'] or
                    wiki_cfg['models'] != d['cfg']['model']):
                raise ValueError('Wiki and DSAA results used different feature/model settings; '
                                 'a shared thesis parameter table would be misleading')
        else:
            log.warning('Wiki results have no recorded run configuration; cannot verify shared parameters')
    d['wikipedia_crawl'] = _load_json_optional(
        pathlib.Path('data/raw/wiki_cs_8k/crawl_stats.json'),
        'run python -m scripts.wiki.build_graph first')
    d['wikipedia_text'] = _load_json_optional(
        pathlib.Path('data/raw/wiki_cs_8k/text_fetch_stats.json'),
        'run python -m scripts.wiki.fetch_text first')

    return d


# ---------------------------------------------------------------------------
# Compute macros (Greek-formatted tex values)
# ---------------------------------------------------------------------------

_SPARSE_MODEL_MACROS = {'structural': 'Structural', 'tfidf': 'Tfidf', 'pos': 'Pos',
                        'embedding': 'Embedding', 'svm': 'Svm'}


def _sparse_macros(sparse_random: dict, sparse_mixed: dict) -> dict[str, str]:
    """Macros for the sparse Wiki stress test (fixed graph/test sets, changed training negatives)."""
    sparse_counts = sparse_random['benchmark']['counts']
    m: dict[str, str] = {}
    m['WsRetentionPct'] = gpct(100 * sparse_random['benchmark']['edge_retention'], 1)
    m['WsObservedEdges'] = gint(sparse_counts['observed_positive_edges'])
    m['WsHeldOutEdges'] = gint(sparse_counts['held_out_positive_edges'])
    m['WsTrainPairs'] = gint(sparse_counts['train_pairs'])
    m['WsTestPairs'] = gint(sparse_counts['test_hard_pairs'])
    m['WsObservedMeanDegree'] = gfloat(sparse_random['benchmark']['observed_graph']['mean_degree'], 1)
    m['WsMixedRandomTrainNegatives'] = gint(
        sparse_mixed['benchmark']['counts']['train_random_negatives'])
    m['WsMixedHardTrainNegatives'] = gint(
        sparse_mixed['benchmark']['counts']['train_hard_negatives'])
    for run, prefix in ((sparse_random, 'WsRandom'), (sparse_mixed, 'WsMixed')):
        for suite, suffix in (('test_random', 'Random'), ('test_hard', 'Hard')):
            results = run['suites'][suite]
            m[f'{prefix}{suffix}CascadeFone'] = gfloat(results['cascade']['macro_f1'], 4)
            m[f'{prefix}{suffix}CascadeAuc'] = gfloat(results['cascade']['auc_roc'], 4)
            for model in _MODEL_NAMES[:-1]:
                m[f'{prefix}{suffix}{_SPARSE_MODEL_MACROS[model]}Fone'] = gfloat(
                    results[model]['macro_f1'], 4)
            best = max(results[model]['macro_f1'] for model in _MODEL_NAMES[:-1])
            m[f'{prefix}{suffix}BestBaselineFone'] = gfloat(best, 4)
            m[f'{prefix}{suffix}CascadeDeltaBest'] = gfloat(results['cascade']['macro_f1'] - best, 4)
            m[f'{prefix}{suffix}TierOnePct'] = gpct(
                results['cascade']['tier_stats']['tier1']['pct'], 1)
            m[f'{prefix}{suffix}TierTwoPct'] = gpct(
                results['cascade']['tier_stats']['tier2']['pct'], 1)
            m[f'{prefix}{suffix}TierThreePct'] = gpct(
                results['cascade']['tier_stats']['tier3']['pct'], 1)
            m[f'{prefix}{suffix}TierOneErrors'] = gint(
                results['cascade']['tier_performance']['tier1']['errors'])
            m[f'{prefix}{suffix}TierOneFone'] = gfloat(
                results['cascade']['tier_performance']['tier1']['macro_f1'], 4)
    random_hard = sparse_random['suites']['test_hard']['cascade']['macro_f1']
    mixed_hard = sparse_mixed['suites']['test_hard']['cascade']['macro_f1']
    random_random = sparse_random['suites']['test_random']['cascade']['macro_f1']
    mixed_random = sparse_mixed['suites']['test_random']['cascade']['macro_f1']
    m['WsHardGain'] = gfloat(mixed_hard - random_hard, 4)
    m['WsRandomCost'] = gfloat(random_random - mixed_random, 4)
    delta, lo, hi = _paired_sparse_hard_delta(sparse_mixed)
    m['WsMixedHardDeltaEmbedding'] = gfloat(delta, 4)
    m['WsMixedHardDeltaEmbeddingCiLow'] = gfloat(lo, 4)
    m['WsMixedHardDeltaEmbeddingCiHigh'] = gfloat(hi, 4)
    m['WsBootstrapResamples'] = gint(SPARSE_BOOTSTRAP_RESAMPLES)
    return m


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

    # ---- Historically named cold-start metric: zero common neighbours ----
    # This preserves the historical DSAA diagnostic. It is broader than
    # operational cold-start because both endpoints may be present in the graph.
    zero_cn = d['struct_train'].loc[vt['id'].to_numpy(), 'cn'].fillna(0).to_numpy() == 0
    cs_count  = int(zero_cn.sum())
    cs_pct    = 100 * cs_count / n_vt
    non_cs    = n_vt - cs_count
    m['ColdStartCount']    = gint(cs_count)
    m['ColdStartPct']      = gfloat(cs_pct, 3)
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
    tt = d['test_tiers']
    n_tt = len(tt)
    tt0_n = int((tt['tier_used'] == 0).sum())
    tt1_n = int((tt['tier_used'] == 1).sum())
    tt2_n = int((tt['tier_used'] == 2).sum())
    tt1_pct = 100 * tt1_n / n_tt
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

    # ---- Equal-training-size DSAA comparison (three matched 20k samples) ----
    matched = d['matched_samples']
    matched_names = {
        'tfidf_lr': 'MatchedTfidfLr',
        'tfidf_svm': 'MatchedTfidfSvm',
        'pos_rf': 'MatchedPos',
        'cascade': 'MatchedCascade',
    }
    if matched is not None:
        required = {'model', 'sample_seed', 'n_train', 'n_val', 'graph_label_budget',
                    'macro_f1', 'auc_roc'}
        missing = required - set(matched.columns)
        if missing:
            raise ValueError(f'Matched-sample metrics lack columns: {sorted(missing)}')
        if set(matched['model']) != set(matched_names):
            raise ValueError('Matched-sample metrics do not contain exactly the expected models')
        if matched[['n_train', 'n_val', 'graph_label_budget']].nunique().max() != 1:
            raise ValueError('Matched-sample rows do not share one training/validation/graph budget')

        m['MatchedSampleSize'] = gint(matched['n_train'].iloc[0])
        m['MatchedValSize'] = gint(matched['n_val'].iloc[0])
        m['MatchedNSeeds'] = str(matched['sample_seed'].nunique())
        for model, prefix in matched_names.items():
            rows = matched[matched['model'] == model]
            m[f'{prefix}Fone'] = gfloat(rows['macro_f1'].mean(), 4)
            m[f'{prefix}FoneStd'] = gfloat(rows['macro_f1'].std(ddof=1), 4)
            m[f'{prefix}Auc'] = gfloat(rows['auc_roc'].mean(), 4)
    else:
        m['MatchedSampleSize'] = m['MatchedValSize'] = m['MatchedNSeeds'] = _MISSING
        for prefix in matched_names.values():
            m[f'{prefix}Fone'] = _MISSING
            m[f'{prefix}FoneStd'] = _MISSING
            m[f'{prefix}Auc'] = _MISSING


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

    # ---- Cached-data protocol audit: graph-construction leakage in the legacy
    # DSAA 2023 row split, and endpoint-order sensitivity of the trained
    # CascadeLP checkpoint (Chapter 4/5 limitations) ----
    pa = d['protocol_audit']
    if pa:
        m['AuditValPositivesInGraph']    = gint(pa['validation_positives_in_graph'])
        m['AuditValN']                   = gint(pa['n_val'])
        m['AuditValPositivesInGraphPct'] = gfloat(100 * pa['validation_positives_in_graph'] / pa['n_val'], 2)
        m['AuditCrossSplitGroups']       = gint(pa['cross_split_unordered_groups'])
        m['AuditFullGraphColdTotal']         = gint(pa['full_graph']['cold_total'])
        m['AuditFullGraphColdPositive']      = gint(pa['full_graph']['cold_positive'])
        m['AuditTrainGraphColdTotal']        = gint(pa['training_partition_graph']['cold_total'])
        m['AuditTrainGraphColdPositive']     = gint(pa['training_partition_graph']['cold_positive'])
        m['AuditTrainGraphColdPositivePct']  = gfloat(
            100 * pa['training_partition_graph']['cold_positive'] / pa['n_val'], 2)
        if 'swap' in pa:
            sw = pa['swap']
            m['SwapN']                = gint(sw['n'])
            m['SwapLabelChangePct']   = gfloat(sw['label_change_pct'], 2)
            m['SwapMeanAbsProbChange'] = gfloat(sw['mean_abs_probability_change'], 3)
            m['SwapTierChanges']      = gint(sw['tier_changes'])
            m['SwapNegClassChangePct'] = gfloat(sw['by_original_class']['0']['label_change_pct'], 2)
            m['SwapPosClassChangePct'] = gfloat(sw['by_original_class']['1']['label_change_pct'], 2)
        else:
            for k in ('SwapN', 'SwapLabelChangePct', 'SwapMeanAbsProbChange', 'SwapTierChanges',
                      'SwapNegClassChangePct', 'SwapPosClassChangePct'):
                m[k] = _MISSING
    else:
        for k in ('AuditValPositivesInGraph', 'AuditValN', 'AuditValPositivesInGraphPct',
                  'AuditCrossSplitGroups', 'AuditFullGraphColdTotal', 'AuditFullGraphColdPositive',
                  'AuditTrainGraphColdTotal', 'AuditTrainGraphColdPositive', 'AuditTrainGraphColdPositivePct',
                  'SwapN', 'SwapLabelChangePct', 'SwapMeanAbsProbChange', 'SwapTierChanges',
                  'SwapNegClassChangePct', 'SwapPosClassChangePct'):
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
        cascade_performance = wf.get('cascade', {}).get('tier_performance', {})
        for t, name in [('tier0', 'Zero'), ('tier1', 'One'), ('tier2', 'Two'), ('tier3', 'Three')]:
            stats = cascade_tiers.get(t)
            performance = cascade_performance.get(t)
            m[f'WfTier{name}Pct'] = gfloat(stats['pct'], 1) if stats else _MISSING
            m[f'WfTier{name}Count'] = gint(stats['n']) if stats else _MISSING
            m[f'WfTier{name}Fone'] = gfloat(performance['macro_f1'], 4) if performance else _MISSING
            m[f'WfTier{name}Errors'] = gint(performance['errors']) if performance else _MISSING

        bootstrap = wf.get('bootstrap', {})
        cascade_ci = bootstrap.get('cascade_macro_f1_ci95')
        delta_ci = bootstrap.get('cascade_minus_structural_ci95')
        m['WfBootstrapResamples'] = gint(bootstrap.get('n_resamples', 0)) if bootstrap else _MISSING
        m['WfCascadeCiLow'] = gfloat(cascade_ci[0], 4) if cascade_ci else _MISSING
        m['WfCascadeCiHigh'] = gfloat(cascade_ci[1], 4) if cascade_ci else _MISSING
        m['WfCascadeDeltaStructural'] = gfloat(
            bootstrap['cascade_minus_structural_macro_f1'], 4) if bootstrap else _MISSING
        m['WfCascadeDeltaStructuralCiLow'] = gfloat(delta_ci[0], 4) if delta_ci else _MISSING
        m['WfCascadeDeltaStructuralCiHigh'] = gfloat(delta_ci[1], 4) if delta_ci else _MISSING

        diagnostics = wf.get('cascade', {}).get('diagnostic_subsets', {})
        for key, name in [('zero_cn', 'ZeroCn'), ('functional_cold_start', 'ColdStart'),
                          ('missing_text', 'MissingText')]:
            diagnostic = diagnostics.get(key)
            m[f'Wf{name}Count'] = gint(diagnostic['n']) if diagnostic else _MISSING
            m[f'Wf{name}Fone'] = (
                gfloat(diagnostic['macro_f1'], 4)
                if diagnostic and diagnostic.get('macro_f1') is not None else _MISSING)
            m[f'Wf{name}PosPct'] = (
                gpct(100 * diagnostic['positive_rate'], 1)
                if diagnostic and diagnostic.get('positive_rate') is not None else _MISSING)
            m[f'Wf{name}Errors'] = gint(diagnostic['errors']) if diagnostic else _MISSING
            m[f'Wf{name}FalseNegatives'] = (
                gint(diagnostic['false_negatives']) if diagnostic else _MISSING)
            m[f'Wf{name}PositiveRecall'] = (
                gfloat(diagnostic['positive_recall'], 4)
                if diagnostic and diagnostic.get('positive_recall') is not None else _MISSING)
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
            m[f'WfTier{name}Fone'] = m[f'WfTier{name}Errors'] = _MISSING
        for k in ('WfBootstrapResamples', 'WfCascadeCiLow', 'WfCascadeCiHigh',
                  'WfCascadeDeltaStructural', 'WfCascadeDeltaStructuralCiLow',
                  'WfCascadeDeltaStructuralCiHigh', 'WfZeroCnCount', 'WfZeroCnFone',
                  'WfColdStartCount', 'WfColdStartFone', 'WfMissingTextCount',
                  'WfMissingTextFone', 'WfZeroCnPosPct', 'WfZeroCnErrors',
                  'WfZeroCnFalseNegatives', 'WfZeroCnPositiveRecall', 'WfColdStartPosPct',
                  'WfColdStartErrors', 'WfColdStartFalseNegatives', 'WfColdStartPositiveRecall',
                  'WfMissingTextPosPct', 'WfMissingTextErrors', 'WfMissingTextFalseNegatives',
                  'WfMissingTextPositiveRecall'):
            m[k] = _MISSING
        for k in ('WfTrainSplitSize', 'WfValSplitSize', 'WfGraphNodes', 'WfGraphEdges', 'WfMeanDegree'):
            m[k] = _MISSING

    # ---- Sparse Wiki stress test: fixed graph/test sets, changed training negatives ----
    m.update(_sparse_macros(d['wiki_sparse_random'], d['wiki_sparse_mixed']))

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
    ref_rows = kg

    kaggle_our_private = None
    if not ref_rows.empty:
        best_ref = ref_rows.sort_values('public_score', ascending=False).iloc[0]
        m['KaggleBaselinePublic']  = gfloat(best_ref['public_score'], 5)
        m['KaggleBaselinePrivate'] = gfloat(best_ref['private_score'], 5)
        kaggle_our_private = float(best_ref['private_score'])
    else:
        m['KaggleBaselinePublic']  = _MISSING
        m['KaggleBaselinePrivate'] = _MISSING

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    d = _load()
    m, shared = _compute_macros(d)
    figs = _compute_figures(d, shared)

    PUBLISHED_RESULTS.mkdir(parents=True, exist_ok=True)
    out = PUBLISHED_RESULTS / 'summary_stats.json'
    out.write_text(json.dumps({'macros': m, 'figures': figs}, indent=2))
    log.info('Wrote %d macros + %d figure groups → %s', len(m), len(figs), out)


if __name__ == '__main__':
    main()
