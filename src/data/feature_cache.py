"""Validate cached feature files against the inputs and settings that made them."""

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cache_matches(directory: Path, request: dict, files: dict[str, Path]) -> bool:
    """Return False for an empty cache; reject partial, stale, or unverified caches."""
    manifest_path = directory / 'feature_cache.json'
    present = {name: path.exists() for name, path in files.items()}
    if not manifest_path.exists() and not any(present.values()):
        return False
    if not manifest_path.exists() or not all(present.values()):
        raise ValueError(
            f'Incomplete or unverified feature cache in {directory}; '
            'use --rebuild-features or a fresh --directory')
    metadata = json.loads(manifest_path.read_text())
    if metadata.get('request') != request:
        raise ValueError(
            f'Feature cache inputs/settings differ in {directory}; '
            'use --rebuild-features or a fresh --directory')
    for name, path in files.items():
        if metadata.get('files', {}).get(name) != sha256_file(path):
            raise ValueError(f'Feature cache checksum mismatch: {path}')
    return True


def save_cache_manifest(directory: Path, request: dict, files: dict[str, Path]) -> None:
    metadata = {
        'request': request,
        'files': {name: sha256_file(path) for name, path in files.items()},
    }
    (directory / 'feature_cache.json').write_text(json.dumps(metadata, indent=2) + '\n')
