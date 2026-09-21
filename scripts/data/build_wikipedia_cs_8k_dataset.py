"""
Turn the crawled real-edge graph (build_from_wikidump.py) into a labeled pair
set: every positive is a genuine hyperlink; negatives are sampled uniformly at
random from the same 8,000-node pool, excluding any real edge. Unlike DSAA
2023, this dataset has no negative-sampling artifact by construction — there
is nothing to correct afterward.

Reads:  <output>/positive_edges.csv, <output>/nodes.tsv
Writes: <output>/train.csv   (id, id1, id2, label — same schema as data/raw/dsaa/train.csv)

Usage:
    python -m scripts.data.build_wikipedia_dataset --output data/raw/wikipedia_cs_8k
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

from src.utils.log_utils import setup_logging

log = setup_logging('build_wikipedia_dataset')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/raw/wikipedia_cs_8k')
    parser.add_argument('--neg-ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out_dir = pathlib.Path(args.output)

    positives = pd.read_csv(out_dir / 'positive_edges.csv')
    nodes = pd.read_csv(out_dir / 'nodes.tsv', sep='\t', index_col='id')
    node_pool = nodes.index.to_numpy()
    log.info('Real positive edges: %s, node pool: %s', f'{len(positives):,}', f'{len(node_pool):,}')

    positive_pairs = set(tuple(sorted((int(u), int(v)))) for u, v in
                         positives[['id1', 'id2']].itertuples(index=False))

    n_neg_target = int(round(len(positives) * args.neg_ratio))
    rng = np.random.default_rng(args.seed)
    negatives = set()
    attempts = 0
    max_attempts = n_neg_target * 30 + 20_000
    while len(negatives) < n_neg_target and attempts < max_attempts:
        batch = n_neg_target - len(negatives)
        u = rng.choice(node_pool, size=batch * 2)
        v = rng.choice(node_pool, size=batch * 2)
        for uu, vv in zip(u.tolist(), v.tolist()):
            if uu == vv:
                continue
            pair = tuple(sorted((int(uu), int(vv))))
            if pair in positive_pairs or pair in negatives:
                continue
            negatives.add(pair)
            if len(negatives) >= n_neg_target:
                break
        attempts += batch * 2
    if len(negatives) < n_neg_target:
        log.warning('Only sampled %d/%d negatives', len(negatives), n_neg_target)

    pos_rows = positives[['id1', 'id2']].copy()
    pos_rows['label'] = 1
    neg_rows = pd.DataFrame(list(negatives), columns=['id1', 'id2'])
    neg_rows['label'] = 0

    dataset = pd.concat([pos_rows, neg_rows], ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    dataset.insert(0, 'id', range(len(dataset)))
    dataset.to_csv(out_dir / 'train.csv', index=False)
    log.info('Wrote %s rows (%s positive, %s negative) -> %s',
             f'{len(dataset):,}', f'{len(pos_rows):,}', f'{len(neg_rows):,}', out_dir / 'train.csv')


if __name__ == '__main__':
    main()
