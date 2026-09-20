"""
Evaluate the original, unmodified CascadeLP (and all four baselines) on the
freshly-crawled, artifact-free Wikipedia dataset (scripts/data/build_from_wikidump.py
+ build_wiki_fresh_dataset.py): a real, connected subgraph of Simple English
Wikipedia, real hyperlinks as positives, honestly-sampled random non-edges as
negatives. Unlike DSAA 2023, there is no known artifact here to correct — this
experiment exists to show what CascadeLP does on a dataset that was built
correctly from the start.

Uses the same grouped, pair-level split and target-edge-masked graph
construction developed for the DSAA 2023 leakage fix (src/data/protocol.py) —
that methodology is good practice regardless of which dataset it's applied to,
and reusing it here avoids reintroducing the exact leak this thesis spent so
much effort diagnosing.

Usage:
    python -m scripts.analysis.run_wiki_fresh_experiment
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.data.loader import build_graph, load_edges, load_nodes_for_ids
from src.data.protocol import prepare_split
from src.features.embeddings import build_tfidf, clean_wiki_text, compute_embedding_scores, compute_tfidf_scores, encode_nodes
from src.features.linguistic import compute_pos_features
from src.features.structural import compute_heuristics
from src.models.cascade import CascadeLP
from src.models.svm import EmbeddingClassifier, PosClassifier, StructuralClassifier, SvmClassifier, TfidfClassifier
from src.utils.log_utils import setup_logging

log = setup_logging('run_wiki_fresh_experiment')


def evaluate(y_true, y_pred, y_score) -> dict:
    return {
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'auc_roc': float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        'n': int(len(y_true)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', default='data/wiki_fresh/train.csv')
    parser.add_argument('--directory', default='data/interim/wiki_fresh')
    parser.add_argument('--nodes', default='data/wiki_fresh/nodes.tsv')
    parser.add_argument('--predictions', default='outputs/predictions/wiki_fresh')
    parser.add_argument('--stats', default='outputs/stats/wiki_fresh_experiment_results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-size', type=float, default=0.2)
    args = parser.parse_args()
    t0 = time.time()

    pairs = load_edges(args.pairs)
    log.info('Loaded %s pairs', f'{len(pairs):,}')

    directory = pathlib.Path(args.directory)
    partition, manifest = prepare_split(pairs, directory, args.val_size, args.seed)
    tr = np.flatnonzero(partition.to_numpy() == 'train')
    val = np.flatnonzero(partition.to_numpy() == 'val')
    y = pairs['label'].to_numpy()
    log.info('Train: %s  Val: %s', f'{len(tr):,}', f'{len(val):,}')

    struct_path, tfidf_path, pos_path, st_path = (
        directory / 'structural.csv', directory / 'tfidf.csv',
        directory / 'pos.npy', directory / 'sentence_emb.csv')

    if struct_path.exists():
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

    if tfidf_path.exists():
        log.info('Loading cached TF-IDF scores')
        tfidf_scores = pd.read_csv(tfidf_path, index_col='id')['tfidf_score'].to_numpy()
    else:
        nodes = load_nodes_for_ids(args.nodes, unique_ids)
        log.info('Fitting TF-IDF and scoring pairs...')
        texts = [clean_wiki_text(nodes.loc[i, 'text']) for i in unique_ids if i in nodes.index]
        vectorizer = build_tfidf(texts)
        tfidf_scores = compute_tfidf_scores(vectorizer, nodes, pairs)
        pd.Series(tfidf_scores, index=pairs.index, name='tfidf_score').to_csv(tfidf_path)

    if pos_path.exists():
        log.info('Loading cached POS features')
        pos_features = np.load(pos_path)
    else:
        if 'nodes' not in dir():
            nodes = load_nodes_for_ids(args.nodes, unique_ids)
        log.info('Computing POS features...')
        pos_features = compute_pos_features(nodes, pairs)
        np.save(pos_path, pos_features)

    if st_path.exists():
        log.info('Loading cached Sentence-Transformer scores')
        st_scores = pd.read_csv(st_path, index_col='id')['st_score'].to_numpy()
    else:
        if 'nodes' not in dir():
            nodes = load_nodes_for_ids(args.nodes, unique_ids)
        log.info('Encoding nodes with Sentence-Transformer...')
        embeddings = encode_nodes(nodes, list(unique_ids))
        st_scores = compute_embedding_scores(embeddings, pairs)
        pd.Series(st_scores, index=pairs.index, name='st_score').to_csv(st_path)

    log.info('Feature computation done (or loaded from cache) at %.1f min', (time.time() - t0) / 60)

    def sub(v, idx):
        return v.iloc[idx] if isinstance(v, pd.DataFrame) else v[idx]

    results = {}
    pred_dir = pathlib.Path(args.predictions)
    pred_dir.mkdir(parents=True, exist_ok=True)

    log.info('Training StructuralClassifier...')
    m = StructuralClassifier(random_state=args.seed)
    m.fit(sub(structural, tr), y[tr])
    proba = m.predict_proba(sub(structural, val))
    results['structural'] = evaluate(y[val], proba.argmax(1), proba[:, 1])

    log.info('Training TfidfClassifier...')
    m = TfidfClassifier(random_state=args.seed)
    m.fit(sub(tfidf_scores, tr), y[tr])
    proba = m.predict_proba(sub(tfidf_scores, val))
    results['tfidf'] = evaluate(y[val], proba.argmax(1), proba[:, 1])

    log.info('Training PosClassifier...')
    m = PosClassifier(random_state=args.seed)
    m.fit(sub(pos_features, tr), y[tr])
    proba = m.predict_proba(sub(pos_features, val))
    results['pos'] = evaluate(y[val], proba.argmax(1), proba[:, 1])

    log.info('Training EmbeddingClassifier...')
    m = EmbeddingClassifier(random_state=args.seed)
    m.fit(sub(st_scores, tr), y[tr])
    proba = m.predict_proba(sub(st_scores, val))
    results['embedding'] = evaluate(y[val], proba.argmax(1), proba[:, 1])

    log.info('Training SvmClassifier...')
    m = SvmClassifier(random_state=args.seed)
    m.fit(sub(tfidf_scores, tr), y[tr])
    proba = m.predict_proba(sub(tfidf_scores, val))
    results['svm'] = evaluate(y[val], proba.argmax(1), proba[:, 1])

    log.info('Training CascadeLP (original, unmodified architecture)...')
    m = CascadeLP()
    m.fit(sub(structural, tr), sub(pos_features, tr), sub(st_scores, tr), y[tr], pairs.iloc[tr])
    y_pred, tier_used, scores = m.predict(sub(structural, val), sub(pos_features, val), sub(st_scores, val), pairs.iloc[val])
    results['cascade'] = evaluate(y[val], y_pred, scores)
    results['cascade']['tier_stats'] = m.tier_stats(tier_used)
    pd.DataFrame({'id': pairs.iloc[val].index, 'y_true': y[val], 'y_pred': y_pred,
                 'tier_used': tier_used, 'score': scores}).to_csv(
        pred_dir / 'cascade_val_tiers.csv', index=False)

    results['_meta'] = {
        'n_train': int(len(tr)), 'n_val': int(len(val)),
        'graph_nodes': graph.number_of_nodes(), 'graph_edges': graph.number_of_edges(),
        'elapsed_min': (time.time() - t0) / 60,
    }

    log.info('=== Results ===')
    for name, r in results.items():
        if name == '_meta':
            continue
        log.info('%-12s Macro F1=%.4f  AUC=%s  n=%d', name, r['macro_f1'],
                 f"{r['auc_roc']:.4f}" if r['auc_roc'] else 'n/a', r['n'])
        if 'tier_stats' in r:
            for tier, stats in r['tier_stats'].items():
                log.info('    %s: %s (%.1f%%)', tier, f"{stats['n']:,}", stats['pct'])

    out = pathlib.Path(args.stats)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info('Saved -> %s', out)


if __name__ == '__main__':
    main()
