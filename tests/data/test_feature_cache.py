import pytest

from src.data.feature_cache import cache_matches, save_cache_manifest


def test_cache_requires_matching_inputs_and_unchanged_files(tmp_path):
    path = tmp_path / 'tfidf.csv'
    files = {'tfidf': path}
    request = {'nodes_sha256': 'nodes-v1', 'model': 'encoder-v1'}
    assert cache_matches(tmp_path, request, files) is False
    path.write_text('id,score\n1,0.5\n')
    with pytest.raises(ValueError, match='unverified'):
        cache_matches(tmp_path, request, files)
    save_cache_manifest(tmp_path, request, files)
    assert cache_matches(tmp_path, request, files) is True
    with pytest.raises(ValueError, match='settings differ'):
        cache_matches(tmp_path, {**request, 'model': 'encoder-v2'}, files)
    path.write_text('id,score\n1,0.7\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        cache_matches(tmp_path, request, files)
