"""
Evaluate baselines and CascadeLP on the Wiki dataset using shared YAML settings.

Reads:
    <pairs>, <nodes>, and <config-dir>/{config,features,training,model} YAML files
Writes:
    <directory>/feature_cache.json and cached features
    <predictions>/model_val_predictions.csv and cascade_val_tiers.csv
    <stats> JSON results with the resolved experiment settings

Usage:
    python -m scripts.wiki.run_experiment [--config-dir configs] [--rebuild-features]
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, roc_auc_score

from src.data.feature_cache import cache_matches, save_cache_manifest, sha256_file
from src.data.loader import build_graph, load_edges, load_nodes_for_ids
from src.data.protocol import prepare_split
from src.features.embeddings import build_tfidf, clean_wiki_text, compute_embedding_scores, compute_tfidf_scores, encode_nodes
from src.features.linguistic import compute_pos_features
from src.features.structural import compute_heuristics
from src.models.cascade import CascadeLP
from src.models.svm import EmbeddingClassifier, PosClassifier, StructuralClassifier, SvmClassifier, TfidfClassifier
from src.utils.log_utils import setup_logging

log = setup_logging('wiki_run_experiment')


def evaluate(y_true, y_pred, y_score) -> dict:
    return {
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'auc_roc': float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        'n': int(len(y_true)),
    }


def bootstrap_comparison(y_true, cascade_pred, structural_pred, seed, n_boot=500) -> dict:
    """Paired bootstrap intervals for CascadeLP and its gain over the strongest baseline."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    cascade_scores = np.empty(n_boot)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        cascade_scores[i] = f1_score(y_true[idx], cascade_pred[idx], average='macro')
        structural_score = f1_score(y_true[idx], structural_pred[idx], average='macro')
        deltas[i] = cascade_scores[i] - structural_score
    return {
        'n_resamples': n_boot,
        'cascade_macro_f1_ci95': np.quantile(cascade_scores, [0.025, 0.975]).tolist(),
        'cascade_minus_structural_macro_f1': float(
            f1_score(y_true, cascade_pred, average='macro')
            - f1_score(y_true, structural_pred, average='macro')),
        'cascade_minus_structural_ci95': np.quantile(deltas, [0.025, 0.975]).tolist(),
    }


def evaluate_subset(y_true, y_pred, y_score, mask) -> dict:
    """Evaluate a diagnostic slice and retain its balance and error direction."""
    truth = y_true[mask]
    prediction = y_pred[mask]
    result = evaluate(truth, prediction, y_score[mask])
    positives = truth == 1
    result.update({
        'positive_rate': float(truth.mean()),
        'errors': int((prediction != truth).sum()),
        'false_negatives': int(((truth == 1) & (prediction == 0)).sum()),
        'false_positives': int(((truth == 0) & (prediction == 1)).sum()),
        'positive_recall': float(prediction[positives].mean()) if positives.any() else None,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', default='data/raw/wiki_cs_8k/train.csv')
    parser.add_argument('--directory', default='data/interim/wiki_cs_8k')
    parser.add_argument('--nodes', default='data/raw/wiki_cs_8k/nodes.tsv')
    parser.add_argument('--predictions', default='outputs/predictions/wiki_cs_8k')
    parser.add_argument('--stats', default='outputs/stats/wiki_cs_8k_experiment_results.json')
    parser.add_argument('--config-dir', default='configs', help='Shared feature/model YAML directory')
    parser.add_argument('--seed', type=int, help='Defaults to config.yaml seed')
    parser.add_argument('--val-size', type=float, help='Defaults to training/default.yaml val_size')
    parser.add_argument('--rebuild-features', action='store_true', help='Replace feature caches and their manifest')
    args = parser.parse_args()
    t0 = time.time()
    config_dir = pathlib.Path(args.config_dir)
    base_cfg = yaml.safe_load((config_dir / 'config.yaml').read_text())
    training_cfg = yaml.safe_load((config_dir / 'training/default.yaml').read_text())
    feature_cfg = yaml.safe_load((config_dir / 'features/default.yaml').read_text())
    model_cfg = {
        name: yaml.safe_load((config_dir / f'model/{name}.yaml').read_text())
        for name in ('structural', 'tfidf', 'pos', 'embedding', 'svm', 'cascade')
    }
    args.seed = base_cfg['seed'] if args.seed is None else args.seed
    args.val_size = training_cfg['val_size'] if args.val_size is None else args.val_size

    pairs = load_edges(args.pairs)
    log.info('Loaded %s pairs', f'{len(pairs):,}')

    directory = pathlib.Path(args.directory)
    partition, manifest = prepare_split(pairs, directory, args.val_size, args.seed)
    tr = np.flatnonzero(partition.to_numpy() == 'train')
    val = np.flatnonzero(partition.to_numpy() == 'val')
    y = pairs['label'].to_numpy()
    log.info('Train: %s  Val: %s', f'{len(tr):,}', f'{len(val):,}')

    files = {
        'structural': directory / 'structural.csv',
        'tfidf': directory / 'tfidf.csv',
        'pos': directory / 'pos.npy',
        'sentence_emb': directory / 'sentence_emb.csv',
    }
    struct_path, tfidf_path, pos_path, st_path = files.values()
    request = {
        'feature_pipeline_version': 1,
        'split': manifest,
        'nodes_sha256': sha256_file(pathlib.Path(args.nodes)),
        'features': feature_cfg,
        'feature_code_sha256': {
            name: sha256_file(pathlib.Path(__file__).resolve().parents[2] / 'src/features' / f'{name}.py')
            for name in ('structural', 'embeddings', 'linguistic')
        },
        'loader_code_sha256': sha256_file(
            pathlib.Path(__file__).resolve().parents[2] / 'src/data/loader.py'),
    }
    reuse = False if args.rebuild_features else cache_matches(directory, request, files)

    if reuse:
        log.info('Loading cached structural features')
        structural = pd.read_csv(struct_path, index_col='id')
        graph = build_graph(pairs.iloc[tr])
    else:
        log.info('Building training-partition graph and target-edge-masked structural heuristics...')
        graph = build_graph(pairs.iloc[tr])
        structural = compute_heuristics(graph, pairs, exclude_target_edge=True)
        structural.to_csv(struct_path)
    log.info('Graph: %s nodes, %s edges, mean degree %.1f', f'{graph.number_of_nodes():,}',
             f'{graph.number_of_edges():,}', 2 * graph.number_of_edges() / max(graph.number_of_nodes(), 1))

    unique_ids = set(pd.unique(pairs[['id1', 'id2']].values.ravel()).tolist())

    if reuse:
        log.info('Loading cached TF-IDF scores')
        tfidf_scores = pd.read_csv(tfidf_path, index_col='id')['tfidf_score'].to_numpy()
    else:
        nodes = load_nodes_for_ids(args.nodes, unique_ids)
        log.info('Fitting TF-IDF and scoring pairs...')
        texts = [clean_wiki_text(nodes.loc[i, 'text']) for i in unique_ids if i in nodes.index]
        vectorizer = build_tfidf(texts, **feature_cfg['tfidf'])
        tfidf_scores = compute_tfidf_scores(vectorizer, nodes, pairs)
        pd.Series(tfidf_scores, index=pairs.index, name='tfidf_score').to_csv(tfidf_path)

    if reuse:
        log.info('Loading cached POS features')
        pos_features = np.load(pos_path)
    else:
        if 'nodes' not in dir():
            nodes = load_nodes_for_ids(args.nodes, unique_ids)
        log.info('Computing POS features...')
        pos_features = compute_pos_features(nodes, pairs)
        np.save(pos_path, pos_features)

    if reuse:
        log.info('Loading cached Sentence-Transformer scores')
        st_scores = pd.read_csv(st_path, index_col='id')['st_score'].to_numpy()
    else:
        if 'nodes' not in dir():
            nodes = load_nodes_for_ids(args.nodes, unique_ids)
        log.info('Encoding nodes with Sentence-Transformer...')
        embeddings = encode_nodes(nodes, list(unique_ids), model_name=feature_cfg['embedding']['model_name'])
        st_scores = compute_embedding_scores(embeddings, pairs)
        pd.Series(st_scores, index=pairs.index, name='st_score').to_csv(st_path)

    if not reuse:
        save_cache_manifest(directory, request, files)

    log.info('Feature computation done (or loaded from cache) at %.1f min', (time.time() - t0) / 60)

    def sub(v, idx):
        return v.iloc[idx] if isinstance(v, pd.DataFrame) else v[idx]

    results = {}
    val_predictions = {}
    pred_dir = pathlib.Path(args.predictions)
    pred_dir.mkdir(parents=True, exist_ok=True)

    log.info('Training StructuralClassifier...')
    m = StructuralClassifier(C=model_cfg['structural']['C'],
                             max_iter=model_cfg['structural']['max_iter'], random_state=args.seed)
    m.fit(sub(structural, tr), y[tr])
    proba = m.predict_proba(sub(structural, val))
    val_predictions['structural'] = proba.argmax(1)
    results['structural'] = evaluate(y[val], val_predictions['structural'], proba[:, 1])

    log.info('Training TfidfClassifier...')
    m = TfidfClassifier(C=model_cfg['tfidf']['C'],
                        max_iter=model_cfg['tfidf']['max_iter'], random_state=args.seed)
    m.fit(sub(tfidf_scores, tr), y[tr])
    proba = m.predict_proba(sub(tfidf_scores, val))
    val_predictions['tfidf'] = proba.argmax(1)
    results['tfidf'] = evaluate(y[val], val_predictions['tfidf'], proba[:, 1])

    log.info('Training PosClassifier...')
    m = PosClassifier(n_estimators=model_cfg['pos']['n_estimators'],
                      class_weight=model_cfg['pos']['class_weight'], random_state=args.seed)
    m.fit(sub(pos_features, tr), y[tr])
    proba = m.predict_proba(sub(pos_features, val))
    val_predictions['pos'] = proba.argmax(1)
    results['pos'] = evaluate(y[val], val_predictions['pos'], proba[:, 1])

    log.info('Training EmbeddingClassifier...')
    m = EmbeddingClassifier(C=model_cfg['embedding']['C'],
                            max_iter=model_cfg['embedding']['max_iter'], random_state=args.seed)
    m.fit(sub(st_scores, tr), y[tr])
    proba = m.predict_proba(sub(st_scores, val))
    val_predictions['embedding'] = proba.argmax(1)
    results['embedding'] = evaluate(y[val], val_predictions['embedding'], proba[:, 1])

    log.info('Training SvmClassifier...')
    m = SvmClassifier(C=model_cfg['svm']['C'], gamma=model_cfg['svm']['gamma'],
                      subsample_size=model_cfg['svm']['subsample_size'], random_state=args.seed)
    m.fit(sub(tfidf_scores, tr), y[tr])
    proba = m.predict_proba(sub(tfidf_scores, val))
    val_predictions['svm'] = proba.argmax(1)
    results['svm'] = evaluate(y[val], val_predictions['svm'], proba[:, 1])

    log.info('Training CascadeLP (original, unmodified architecture)...')
    m = CascadeLP(tier1_threshold=model_cfg['cascade']['tier1_threshold'],
                  tier2_threshold=model_cfg['cascade']['tier2_threshold'], random_state=args.seed)
    m.fit(sub(structural, tr), sub(pos_features, tr), sub(st_scores, tr), y[tr], pairs.iloc[tr])
    y_pred, tier_used, scores = m.predict(sub(structural, val), sub(pos_features, val), sub(st_scores, val), pairs.iloc[val])
    results['cascade'] = evaluate(y[val], y_pred, scores)
    val_predictions['cascade'] = y_pred
    results['cascade']['tier_stats'] = m.tier_stats(tier_used)
    results['cascade']['tier_performance'] = {
        f'tier{tier}': {
            **evaluate(y[val][tier_used == tier], y_pred[tier_used == tier],
                       scores[tier_used == tier]),
            'positive_rate': float(y[val][tier_used == tier].mean()),
            'errors': int((y_pred[tier_used == tier] != y[val][tier_used == tier]).sum()),
        }
        for tier in range(4) if (tier_used == tier).any()
    }

    structural_val = sub(structural, val)
    zero_cn = structural_val['cn'].fillna(0).to_numpy() == 0
    heuristic_cols = ['cn', 'jaccard', 'adamic_adar', 'pref_attach']
    functional_cold_start = (structural_val[heuristic_cols].fillna(0).to_numpy() == 0).all(axis=1)
    results['cascade']['diagnostic_subsets'] = {
        'zero_cn': evaluate_subset(y[val], y_pred, scores, zero_cn),
        'functional_cold_start': evaluate_subset(
            y[val], y_pred, scores, functional_cold_start),
    }

    nodes_for_audit = load_nodes_for_ids(args.nodes, unique_ids)
    empty_ids = unique_ids - set(nodes_for_audit.index)
    empty_ids.update(nodes_for_audit.index[
        nodes_for_audit['text'].fillna('').astype(str).str.strip().eq('')])
    val_pairs = pairs.iloc[val]
    missing_text = (val_pairs['id1'].isin(empty_ids) | val_pairs['id2'].isin(empty_ids)).to_numpy()
    results['cascade']['diagnostic_subsets']['missing_text'] = (
        evaluate_subset(y[val], y_pred, scores, missing_text)
        if missing_text.any() else {'macro_f1': None, 'auc_roc': None, 'n': 0})

    results['bootstrap'] = bootstrap_comparison(
        y[val], y_pred, val_predictions['structural'], args.seed)

    pd.DataFrame({'id': pairs.iloc[val].index, 'y_true': y[val], 'y_pred': y_pred,
                 'tier_used': tier_used, 'score': scores}).to_csv(
        pred_dir / 'cascade_val_tiers.csv', index=False)
    prediction_frame = pd.DataFrame({'id': pairs.iloc[val].index, 'y_true': y[val]})
    for name, prediction in val_predictions.items():
        prediction_frame[f'{name}_pred'] = prediction
    prediction_frame.to_csv(pred_dir / 'model_val_predictions.csv', index=False)

    results['_meta'] = {
        'n_train': int(len(tr)), 'n_val': int(len(val)),
        'graph_nodes': graph.number_of_nodes(), 'graph_edges': graph.number_of_edges(),
        'elapsed_min': (time.time() - t0) / 60,
        'run_config': {'seed': args.seed, 'val_size': args.val_size,
                       'features': feature_cfg, 'models': model_cfg},
        'feature_cache_sha256': sha256_file(directory / 'feature_cache.json'),
    }

    log.info('=== Results ===')
    for name, r in results.items():
        if name in ('_meta', 'bootstrap'):
            continue
        log.info('%-12s Macro F1=%.4f  AUC=%s  n=%d', name, r['macro_f1'],
                 f"{r['auc_roc']:.4f}" if r['auc_roc'] else 'n/a', r['n'])
        if 'tier_stats' in r:
            for tier, stats in r['tier_stats'].items():
                log.info('    %s: %s (%.1f%%)', tier, f"{stats['n']:,}", stats['pct'])

    out = pathlib.Path(args.stats)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + '\n')
    log.info('Saved -> %s', out)


if __name__ == '__main__':
    main()
