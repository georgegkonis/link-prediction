"""Resolve configured data paths and record the settings of completed runs."""

import json
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def dsaa_feature_paths(cfg: DictConfig) -> tuple[Path, Path]:
    """Keep limited or skipped development features away from full-data files."""
    raw = Path(to_absolute_path(cfg.paths.raw))
    interim = Path(to_absolute_path(cfg.paths.interim))
    suffix = []
    if cfg.dev.nrows is not None:
        if cfg.dev.nrows < 1:
            raise ValueError('dev.nrows must be a positive integer')
        suffix.append(f'nrows_{cfg.dev.nrows}')
    if cfg.dev.skip_st:
        suffix.append('skip_st')
    if cfg.dev.skip_pos:
        suffix.append('skip_pos')
    if suffix:
        interim /= 'dev_' + '_'.join(suffix)
    interim.mkdir(parents=True, exist_ok=True)
    return raw, interim


def save_run_config(cfg: DictConfig, path: Path, **details) -> None:
    """Write the resolved Hydra configuration next to the result it describes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'config': OmegaConf.to_container(cfg, resolve=True), **details}
    path.write_text(json.dumps(data, indent=2) + '\n')


def load_model_run(predictions: Path, model: str) -> dict | None:
    """Read a completed model's settings, or return None for legacy results."""
    path = predictions / f'{model}_run_config.json'
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    if run['config']['model']['name'] != model:
        raise ValueError(f'Run configuration does not describe {model}: {path}')
    return run
