"""
Compute structural features for train and test pairs.

Reads:
    data/raw/dsaa/train.csv
    data/raw/dsaa/test.csv
Writes:
    data/interim/dsaa/structural_train.csv
    data/interim/dsaa/structural_test.csv

Usage:
    python -m scripts.data.compute_structural [dev.nrows=N]
"""

import hydra
from omegaconf import DictConfig

import numpy as np

from src.data.loader import build_graph, load_edges
from src.features.structural import compute_heuristics
from src.utils.log_utils import setup_logging

INTERIM = 'data/interim/dsaa'


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    log = setup_logging('compute_structural')
    log.info('Loading edges...')
    train = load_edges('data/raw/dsaa/train.csv', nrows=cfg.dev.nrows)
    test  = load_edges('data/raw/dsaa/test.csv',  nrows=cfg.dev.nrows)

    log.info('Building graph from positive training edges...')
    G = build_graph(train)
    log.info('  %s nodes  |  %s edges', f'{G.number_of_nodes():,}', f'{G.number_of_edges():,}')

    log.info('Computing heuristics for train pairs...')
    h_train = compute_heuristics(G, train)
    h_train.to_csv(f'{INTERIM}/structural_train.csv')
    log.info('  Saved → %s/structural_train.csv', INTERIM)

    log.info('Computing heuristics for test pairs...')
    h_test = compute_heuristics(G, test)
    h_test.to_csv(f'{INTERIM}/structural_test.csv')
    log.info('  Saved → %s/structural_test.csv', INTERIM)


    log.info('Done.')


if __name__ == '__main__':
    main()
