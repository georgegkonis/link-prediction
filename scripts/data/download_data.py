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

load_dotenv()

RAW = Path('data/raw')


def main():
    print('Downloading DSAA 2023 competition dataset...')
    cache_path = Path(kagglehub.competition_download('dsaa-2023-competition'))
    print(f'Downloaded to cache: {cache_path}')

    files = list(cache_path.rglob('*'))
    print(f'Files in download: {[f.name for f in files if f.is_file()]}')

    for src in files:
        if not src.is_file():
            continue

        dest = RAW / src.name

        if src.suffix == '.zip':
            print(f'Extracting {src.name} → {RAW}/')
            with zipfile.ZipFile(src) as zf:
                zf.extractall(RAW)
        else:
            print(f'Copying {src.name} → {dest}')
            shutil.copy2(src, dest)

    print(f'\nDone. Files in data/raw/:')
    for f in sorted(RAW.iterdir()):
        print(f'  {f.name}  ({f.stat().st_size / 1024:.1f} KB)')


if __name__ == '__main__':
    main()
