"""
Compute semantic features (TF-IDF cosine similarity, Sentence-Transformer
cosine similarity, POS frequency vectors) for train and test pairs.

Outputs:
    data/interim/tfidf_train.csv        — TF-IDF cosine scores, train pairs
    data/interim/tfidf_test.csv         — TF-IDF cosine scores, test pairs
    data/interim/sentence_emb_train.csv — Sentence-Transformer scores, train
    data/interim/sentence_emb_test.csv  — Sentence-Transformer scores, test
    data/interim/pos_train.npy          — POS feature matrix, train pairs
    data/interim/pos_test.npy           — POS feature matrix, test pairs

Usage:
    python -m scripts.compute_semantic [--nrows N] [--model MODEL] [--skip-st] [--skip-pos]
"""

import argparse

import numpy as np
import pandas as pd

from src.data.loader import load_edges, load_nodes_for_ids
from src.features.embeddings import (
    build_tfidf,
    clean_wiki_text,
    compute_embedding_scores,
    compute_tfidf_scores,
    encode_nodes,
)
from src.features.linguistic import compute_pos_features
from src.utils.log_utils import setup_logging

INTERIM = 'data/interim'


def main(nrows: int | None, model_name: str, skip_st: bool, skip_pos: bool):
    log = setup_logging('compute_semantic')
    log.info('Loading edges...')
    train = load_edges('data/raw/train.csv', nrows=nrows)
    test  = load_edges('data/raw/test.csv',  nrows=nrows)

    all_pairs = pd.concat([train, test])
    unique_ids = set(pd.unique(all_pairs[['id1', 'id2']].values.ravel()).tolist())
    log.info('Unique node IDs across train+test: %s', f'{len(unique_ids):,}')

    log.info('Loading nodes for required IDs (streaming)...')
    nodes = load_nodes_for_ids('data/raw/nodes.tsv', unique_ids)
    log.info('  Loaded %s nodes', f'{len(nodes):,}')

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    log.info('Building TF-IDF vectorizer...')
    texts = [clean_wiki_text(nodes.loc[i, 'text']) for i in unique_ids if i in nodes.index]
    vectorizer = build_tfidf(texts)

    log.info('Computing TF-IDF scores for train pairs...')
    tfidf_train = compute_tfidf_scores(vectorizer, nodes, train)
    pd.Series(tfidf_train, index=train.index, name='tfidf_score').to_csv(f'{INTERIM}/tfidf_train.csv')

    log.info('Computing TF-IDF scores for test pairs...')
    tfidf_test = compute_tfidf_scores(vectorizer, nodes, test)
    pd.Series(tfidf_test, index=test.index, name='tfidf_score').to_csv(f'{INTERIM}/tfidf_test.csv')
    log.info('  Saved → %s/tfidf_{train,test}.csv', INTERIM)

    # ── Sentence Transformers ─────────────────────────────────────────────────
    if not skip_st:
        log.info('Encoding nodes with %s...', model_name)
        embeddings = encode_nodes(nodes, unique_ids, model_name=model_name)

        log.info('Computing embedding scores for train pairs...')
        st_train = compute_embedding_scores(embeddings, train)
        pd.Series(st_train, index=train.index, name='st_score').to_csv(f'{INTERIM}/sentence_emb_train.csv')

        log.info('Computing embedding scores for test pairs...')
        st_test = compute_embedding_scores(embeddings, test)
        pd.Series(st_test, index=test.index, name='st_score').to_csv(f'{INTERIM}/sentence_emb_test.csv')
        log.info('  Saved → %s/sentence_emb_{train,test}.csv', INTERIM)

    # ── POS features ──────────────────────────────────────────────────────────
    if not skip_pos:
        log.info('Computing POS features for train pairs...')
        pos_train = compute_pos_features(nodes, train)
        np.save(f'{INTERIM}/pos_train.npy', pos_train)

        log.info('Computing POS features for test pairs...')
        pos_test = compute_pos_features(nodes, test)
        np.save(f'{INTERIM}/pos_test.npy', pos_test)
        log.info('  Saved → %s/pos_{train,test}.npy', INTERIM)

    log.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrows',    type=int, default=None)
    parser.add_argument('--model',    default='all-MiniLM-L6-v2', help='Sentence-Transformer model name')
    parser.add_argument('--skip-st',  action='store_true', help='Skip sentence transformer encoding')
    parser.add_argument('--skip-pos', action='store_true', help='Skip POS feature computation')
    args = parser.parse_args()
    main(args.nrows, args.model, args.skip_st, args.skip_pos)
