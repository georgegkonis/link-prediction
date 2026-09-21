"""
Download the DSAA 2023 competition dataset from Kaggle and place files in data/raw/.

Requires KAGGLE_USERNAME and KAGGLE_KEY in .env (or already exported in the shell).

Usage:
    python -m scripts.download_data
"""

import os
import shutil
import zipfile
from pathlib import Path

from dotenv import load_dotenv
import kagglehub

from src.utils.log_utils import setup_logging

load_dotenv()

RAW = Path('data/raw/dsaa')
log = setup_logging('download_data')


def main():
    log.info('Downloading DSAA 2023 competition dataset...')
    cache_path = Path(kagglehub.competition_download('dsaa-2023-competition'))
    log.info('Downloaded to cache: %s', cache_path)

    files = list(cache_path.rglob('*'))
    log.info('Files in download: %s', [f.name for f in files if f.is_file()])

    for src in files:
        if not src.is_file():
            continue

        dest = RAW / src.name

        if src.suffix == '.zip':
            log.info('Extracting %s → %s/', src.name, RAW)
            with zipfile.ZipFile(src) as zf:
                zf.extractall(RAW)
        else:
            log.info('Copying %s → %s', src.name, dest)
            shutil.copy2(src, dest)

    log.info('Done. Files in data/raw/dsaa/:')
    for f in sorted(RAW.iterdir()):
        log.info('  %s  (%.1f KB)', f.name, f.stat().st_size / 1024)


if __name__ == '__main__':
    main()
