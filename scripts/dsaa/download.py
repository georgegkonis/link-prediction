"""
Download the configured DSAA 2023 competition dataset from Kaggle.

Reads:
    Kaggle competition files selected by kaggle.competition
Writes:
    data/raw/dsaa/

Usage:
    python -m scripts.dsaa.download [paths.raw=PATH] [kaggle.competition=NAME]
"""

import shutil
import zipfile
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
import kagglehub
from omegaconf import DictConfig

from src.utils.log_utils import setup_logging

load_dotenv()

log = setup_logging('download')


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    raw = Path(to_absolute_path(cfg.paths.raw))
    raw.mkdir(parents=True, exist_ok=True)
    log.info('Downloading DSAA 2023 competition dataset...')
    cache_path = Path(kagglehub.competition_download(cfg.kaggle.competition))
    log.info('Downloaded to cache: %s', cache_path)

    files = list(cache_path.rglob('*'))
    log.info('Files in download: %s', [f.name for f in files if f.is_file()])

    for src in files:
        if not src.is_file():
            continue

        dest = raw / src.name

        if src.suffix == '.zip':
            log.info('Extracting %s → %s/', src.name, raw)
            with zipfile.ZipFile(src) as zf:
                zf.extractall(raw)
        else:
            log.info('Copying %s → %s', src.name, dest)
            shutil.copy2(src, dest)

    log.info('Done. Files in %s/:', raw)
    for f in sorted(raw.iterdir()):
        log.info('  %s  (%.1f KB)', f.name, f.stat().st_size / 1024)


if __name__ == '__main__':
    main()
