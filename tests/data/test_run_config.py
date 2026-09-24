import json

import pytest
from omegaconf import OmegaConf

from src.data.run_config import dsaa_feature_paths, load_model_run, save_run_config


def test_limited_feature_run_uses_separate_output_directory(tmp_path):
    cfg = OmegaConf.create({
        'paths': {'raw': str(tmp_path / 'raw'), 'interim': str(tmp_path / 'interim')},
        'dev': {'nrows': 300, 'skip_st': False, 'skip_pos': False},
    })
    raw, interim = dsaa_feature_paths(cfg)
    assert raw == tmp_path / 'raw'
    assert interim == tmp_path / 'interim' / 'dev_nrows_300'
    assert interim.is_dir()
    save_run_config(cfg, interim / 'semantic_config.json', train_rows=300)
    saved = json.loads((interim / 'semantic_config.json').read_text())
    assert saved['config']['dev']['nrows'] == saved['train_rows'] == 300
    assert not (tmp_path / 'interim' / 'semantic_config.json').exists()


def test_invalid_development_limit_is_rejected(tmp_path):
    cfg = OmegaConf.create({
        'paths': {'raw': str(tmp_path / 'raw'), 'interim': str(tmp_path / 'interim')},
        'dev': {'nrows': 0, 'skip_st': False, 'skip_pos': False},
    })
    with pytest.raises(ValueError, match='positive integer'):
        dsaa_feature_paths(cfg)


def test_skipped_feature_family_is_isolated(tmp_path):
    cfg = OmegaConf.create({
        'paths': {'raw': str(tmp_path / 'raw'), 'interim': str(tmp_path / 'interim')},
        'dev': {'nrows': None, 'skip_st': True, 'skip_pos': False},
    })
    _, interim = dsaa_feature_paths(cfg)
    assert interim == tmp_path / 'interim' / 'dev_skip_st'


def test_model_run_snapshot_is_checked_before_reuse(tmp_path):
    cfg = OmegaConf.create({'model': {'name': 'cascade'}, 'seed': 17})
    assert load_model_run(tmp_path, 'cascade') is None
    save_run_config(cfg, tmp_path / 'cascade_run_config.json', checkpoint='cascade.joblib')
    assert load_model_run(tmp_path, 'cascade')['config']['seed'] == 17
    (tmp_path / 'structural_run_config.json').write_text(
        (tmp_path / 'cascade_run_config.json').read_text())
    with pytest.raises(ValueError, match='does not describe'):
        load_model_run(tmp_path, 'structural')
