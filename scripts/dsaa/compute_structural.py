"""
Compute structural features for train and test pairs.

Reads:
    data/raw/dsaa/train.csv
    data/raw/dsaa/test.csv
Writes:
    data/interim/dsaa/structural_train.csv
    data/interim/dsaa/structural_test.csv
    data/interim/dsaa/structural_config.json

Usage:
    python -m scripts.dsaa.compute_structural [dev.nrows=N]

The paths come from configs/config.yaml. Development limits or skipped feature
families write under <paths.interim>/dev_.../ instead of full-data features.
"""

import hydra
from omegaconf import DictConfig

from src.data.run_config import dsaa_feature_paths, save_run_config
from src.data.loader import build_graph, load_edges
from src.features.structural import compute_heuristics
from src.utils.log_utils import setup_logging

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    log = setup_logging('compute_structural')
    raw, interim = dsaa_feature_paths(cfg)
    log.info('Loading edges...')
    train = load_edges(raw / 'train.csv', nrows=cfg.dev.nrows)
    test = load_edges(raw / 'test.csv', nrows=cfg.dev.nrows)

    log.info('Building graph from positive training edges...')
    G = build_graph(train)
    log.info('  %s nodes  |  %s edges', f'{G.number_of_nodes():,}', f'{G.number_of_edges():,}')

    log.info('Computing heuristics for train pairs...')
    h_train = compute_heuristics(G, train)
    h_train.to_csv(interim / 'structural_train.csv')
    log.info('  Saved → %s/structural_train.csv', interim)

    log.info('Computing heuristics for test pairs...')
    h_test = compute_heuristics(G, test)
    h_test.to_csv(interim / 'structural_test.csv')
    log.info('  Saved → %s/structural_test.csv', interim)
    save_run_config(cfg, interim / 'structural_config.json',
                    train_rows=len(train), test_rows=len(test))
    log.info('Done.')


if __name__ == '__main__':
    main()
