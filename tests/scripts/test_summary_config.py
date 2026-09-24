import json

import pytest

from scripts.thesis import compute_summary_stats as summary


def test_summary_rejects_partial_run_snapshots(tmp_path, monkeypatch):
    predictions = tmp_path / 'predictions'
    predictions.mkdir()
    (predictions / 'cascade_run_config.json').write_text('{}')
    monkeypatch.setattr(summary, 'PREDICTIONS', predictions)
    monkeypatch.setattr(summary, 'INTERIM', tmp_path / 'interim')
    with pytest.raises(ValueError, match='Incomplete run configuration'):
        summary._load_configs()


def test_summary_uses_recorded_settings_instead_of_yaml_defaults(tmp_path, monkeypatch):
    raw, interim, predictions = (tmp_path / name for name in ('raw', 'interim', 'predictions'))
    interim.mkdir()
    predictions.mkdir()
    monkeypatch.setattr(summary, 'RAW', raw)
    monkeypatch.setattr(summary, 'INTERIM', interim)
    monkeypatch.setattr(summary, 'PREDICTIONS', predictions)
    defaults = summary._load_configs()
    paths = {'raw': str(raw), 'interim': str(interim), 'predictions': str(predictions)}
    for name in summary._MODEL_NAMES:
        model = dict(defaults['model'][name])
        if name == 'cascade':
            model['tier1_threshold'] = 0.9
        run = {'config': {'model': model, 'seed': 17,
                          'training': defaults['training'], 'paths': paths},
               'raw_path': str(raw), 'interim_path': str(interim)}
        (predictions / f'{name}_run_config.json').write_text(json.dumps(run))
    features = dict(defaults['features'])
    features['tfidf'] = {**features['tfidf'], 'max_features': 123}
    feature_run = {'config': {'features': features,
                              'dev': {'nrows': None, 'skip_st': False, 'skip_pos': False},
                              'paths': paths}}
    for name in ('structural', 'semantic'):
        (interim / f'{name}_config.json').write_text(json.dumps(feature_run))
    actual = summary._load_configs()
    assert actual['seed'] == 17
    assert actual['model']['cascade']['tier1_threshold'] == 0.9
    assert actual['features']['tfidf']['max_features'] == 123
