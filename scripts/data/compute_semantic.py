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

INTERIM = 'data/interim'


def main(nrows: int | None, model_name: str, skip_st: bool, skip_pos: bool):
    print('Loading edges...')
    train = load_edges('data/raw/train.csv', nrows=nrows)
    test  = load_edges('data/raw/test.csv',  nrows=nrows)

    all_pairs = pd.concat([train, test])
    unique_ids = set(pd.unique(all_pairs[['id1', 'id2']].values.ravel()).tolist())
    print(f'Unique node IDs across train+test: {len(unique_ids):,}')

    print('\nLoading nodes for required IDs (streaming)...')
    nodes = load_nodes_for_ids('data/raw/nodes.tsv', unique_ids)
    print(f'  Loaded {len(nodes):,} nodes')

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    print('\nBuilding TF-IDF vectorizer...')
    texts = [clean_wiki_text(nodes.loc[i, 'text']) for i in unique_ids if i in nodes.index]
    vectorizer = build_tfidf(texts)

    print('Computing TF-IDF scores for train pairs...')
    tfidf_train = compute_tfidf_scores(vectorizer, nodes, train)
    pd.Series(tfidf_train, index=train.index, name='tfidf_score').to_csv(f'{INTERIM}/tfidf_train.csv')

    print('Computing TF-IDF scores for test pairs...')
    tfidf_test = compute_tfidf_scores(vectorizer, nodes, test)
    pd.Series(tfidf_test, index=test.index, name='tfidf_score').to_csv(f'{INTERIM}/tfidf_test.csv')
    print(f'  Saved → {INTERIM}/tfidf_{{train,test}}.csv')

    # ── Sentence Transformers ─────────────────────────────────────────────────
    if not skip_st:
        print(f'\nEncoding nodes with {model_name}...')
        embeddings = encode_nodes(nodes, unique_ids, model_name=model_name)

        print('Computing embedding scores for train pairs...')
        st_train = compute_embedding_scores(embeddings, train)
        pd.Series(st_train, index=train.index, name='st_score').to_csv(f'{INTERIM}/sentence_emb_train.csv')

        print('Computing embedding scores for test pairs...')
        st_test = compute_embedding_scores(embeddings, test)
        pd.Series(st_test, index=test.index, name='st_score').to_csv(f'{INTERIM}/sentence_emb_test.csv')
        print(f'  Saved → {INTERIM}/sentence_emb_{{train,test}}.csv')

    # ── POS features ──────────────────────────────────────────────────────────
    if not skip_pos:
        print('\nComputing POS features for train pairs...')
        pos_train = compute_pos_features(nodes, train)
        np.save(f'{INTERIM}/pos_train.npy', pos_train)

        print('Computing POS features for test pairs...')
        pos_test = compute_pos_features(nodes, test)
        np.save(f'{INTERIM}/pos_test.npy', pos_test)
        print(f'  Saved → {INTERIM}/pos_{{train,test}}.npy')

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrows',    type=int, default=None)
    parser.add_argument('--model',    default='all-MiniLM-L6-v2', help='Sentence-Transformer model name')
    parser.add_argument('--skip-st',  action='store_true', help='Skip sentence transformer encoding')
    parser.add_argument('--skip-pos', action='store_true', help='Skip POS feature computation')
    args = parser.parse_args()
    main(args.nrows, args.model, args.skip_st, args.skip_pos)
