"""
Compute structural features for train and test pairs.

Outputs:
    data/interim/structural_train.csv   — heuristic scores for training pairs
    data/interim/structural_test.csv    — heuristic scores for test pairs
    data/interim/node2vec.kv            — trained Node2Vec KeyedVectors
    data/interim/n2v_train.npy          — 64-dim Node2Vec Hadamard features for training pairs
    data/interim/n2v_test.npy           — 64-dim Node2Vec Hadamard features for test pairs

Usage:
    python -m scripts.compute_structural [--nrows N] [--skip-n2v]
"""

import argparse

import numpy as np
import pandas as pd

from src.data.loader import build_graph, load_edges
from src.features.structural import compute_heuristics, node2vec_hadamard_features, train_node2vec

INTERIM = 'data/interim'


def main(nrows: int | None, skip_n2v: bool):
    print('Loading edges...')
    train = load_edges('data/raw/train.csv', nrows=nrows)
    test  = load_edges('data/raw/test.csv',  nrows=nrows)

    print('Building graph from positive training edges...')
    G = build_graph(train)
    print(f'  {G.number_of_nodes():,} nodes  |  {G.number_of_edges():,} edges')

    print('\nComputing heuristics for train pairs...')
    h_train = compute_heuristics(G, train)
    h_train.to_csv(f'{INTERIM}/structural_train.csv')
    print(f'  Saved → {INTERIM}/structural_train.csv')

    print('\nComputing heuristics for test pairs...')
    h_test = compute_heuristics(G, test)
    h_test.to_csv(f'{INTERIM}/structural_test.csv')
    print(f'  Saved → {INTERIM}/structural_test.csv')

    if not skip_n2v:
        print('\nTraining Node2Vec...')
        wv = train_node2vec(G)
        wv.save(f'{INTERIM}/node2vec.kv')
        print(f'  Saved → {INTERIM}/node2vec.kv')

        print('\nComputing Node2Vec Hadamard features for train pairs...')
        np.save(f'{INTERIM}/n2v_train.npy', node2vec_hadamard_features(wv, train))
        print(f'  Saved → {INTERIM}/n2v_train.npy')

        print('Computing Node2Vec Hadamard features for test pairs...')
        np.save(f'{INTERIM}/n2v_test.npy', node2vec_hadamard_features(wv, test))
        print(f'  Saved → {INTERIM}/n2v_test.npy')

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrows', type=int, default=None, help='Limit rows for fast dev iteration')
    parser.add_argument('--skip-n2v', action='store_true', help='Skip Node2Vec (heuristics only)')
    args = parser.parse_args()
    main(args.nrows, args.skip_n2v)
