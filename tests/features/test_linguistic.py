"""Tests for src/features/linguistic.py.

The NLTK tagger data (`punkt_tab`, `averaged_perceptron_tagger_eng`) is a small
local download; `ensure_nltk_data()` fetches it on demand. If it is unavailable
and cannot be fetched, the whole module is skipped rather than failing.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.linguistic import (
    _POS_TAGS,
    _TAG_INDEX,
    compute_pos_features,
    pos_frequency_vector,
)

N_TAGS = 36


@pytest.fixture(scope='module', autouse=True)
def _require_nltk_data():
    import nltk
    from src.features.linguistic import ensure_nltk_data
    try:
        ensure_nltk_data()
        nltk.pos_tag(nltk.word_tokenize('a test'))
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f'NLTK tagger data unavailable: {exc}')


# ── tag table ────────────────────────────────────────────────────────────────

def test_pos_tag_table_is_36_unique_tags():
    assert len(_POS_TAGS) == N_TAGS
    assert len(set(_POS_TAGS)) == N_TAGS
    assert _TAG_INDEX == {t: i for i, t in enumerate(_POS_TAGS)}


# ── pos_frequency_vector ─────────────────────────────────────────────────────

def test_pos_frequency_vector_all_determiners():
    """'the the the' tags as DT DT DT → mass 1.0 concentrated in the DT slot."""
    vec = pos_frequency_vector('the the the')
    assert vec.shape == (N_TAGS,)
    assert vec[_TAG_INDEX['DT']] == pytest.approx(1.0)
    assert vec.sum() == pytest.approx(1.0)
    assert np.count_nonzero(vec) == 1


@pytest.mark.parametrize('text', ['', '   ', '{{Infobox only}}', '<!-- nothing -->'])
def test_pos_frequency_vector_empty_after_cleaning_is_zero_vector(text):
    vec = pos_frequency_vector(text)
    assert vec.shape == (N_TAGS,)
    assert not vec.any()


@pytest.mark.parametrize('bad', [None, float('nan'), 12])
def test_pos_frequency_vector_non_string_is_zero_vector(bad):
    assert not pos_frequency_vector(bad).any()


def test_pos_frequency_vector_normalises_over_all_tokens_including_untracked():
    """Documented behaviour gap: the denominator is *all* tokens, but only the 36
    tracked tags contribute mass. Punctuation is tagged (`.`, `,`, `:`) and is not
    in the table, so real prose yields a vector summing to strictly less than 1.
    """
    vec = pos_frequency_vector('The quick brown fox jumps over the lazy dog .')
    # 10 tokens, 9 of which carry a tracked tag → 0.9
    assert vec.sum() == pytest.approx(0.9)
    assert vec.sum() < 1.0


def test_pos_frequency_vector_punctuation_only_is_zero_despite_tokens():
    """'...' tokenises to one token tagged ':' which is untracked → all zeros.
    Indistinguishable from empty input downstream."""
    assert not pos_frequency_vector('...').any()


def test_pos_frequency_vector_is_length_invariant_for_repeated_sentences():
    """Frequencies, not counts: repeating the same sentence must not change the
    distribution."""
    a = pos_frequency_vector('The dog runs .')
    b = pos_frequency_vector('The dog runs . The dog runs .')
    np.testing.assert_allclose(a, b)


def test_pos_frequency_vector_values_are_fractions():
    vec = pos_frequency_vector('the dog chased the cat')
    assert ((vec >= 0) & (vec <= 1)).all()
    assert vec[_TAG_INDEX['DT']] == pytest.approx(2 / 5)


def test_pos_frequency_vector_cleans_markup_first():
    assert pos_frequency_vector("{{tpl}} '''the''' the [[X|the]]").tolist() == \
        pytest.approx(pos_frequency_vector('the the the').tolist())


# ── compute_pos_features ─────────────────────────────────────────────────────

@pytest.fixture
def pos_nodes() -> pd.DataFrame:
    df = pd.DataFrame({'text': {
        1: 'the the the',
        2: 'dog cat fox',
        3: '',
    }})
    df.index.name = 'id'
    return df


def test_compute_pos_features_shape_is_72(pos_nodes):
    pairs = pd.DataFrame({'id1': [1, 2], 'id2': [2, 3]})
    feats = compute_pos_features(pos_nodes, pairs, show_progress=False)
    assert feats.shape == (2, 2 * N_TAGS)


def test_compute_pos_features_concatenation_order_is_id1_then_id2(pos_nodes):
    pairs = pd.DataFrame({'id1': [1], 'id2': [2]}, index=[50])
    feats = compute_pos_features(pos_nodes, pairs, show_progress=False)
    v1 = pos_frequency_vector('the the the')
    v2 = pos_frequency_vector('dog cat fox')
    assert feats[0, :N_TAGS].tolist() == pytest.approx(v1.tolist())
    assert feats[0, N_TAGS:].tolist() == pytest.approx(v2.tolist())
    assert feats[0, :N_TAGS].tolist() != pytest.approx(feats[0, N_TAGS:].tolist())


def test_compute_pos_features_not_symmetric_in_pair_order(pos_nodes):
    fwd = compute_pos_features(pos_nodes, pd.DataFrame({'id1': [1], 'id2': [2]}),
                               show_progress=False)
    rev = compute_pos_features(pos_nodes, pd.DataFrame({'id1': [2], 'id2': [1]}),
                               show_progress=False)
    assert fwd[0, :N_TAGS].tolist() == pytest.approx(rev[0, N_TAGS:].tolist())
    assert fwd[0, N_TAGS:].tolist() == pytest.approx(rev[0, :N_TAGS].tolist())


def test_compute_pos_features_missing_node_gets_zero_half(pos_nodes):
    pairs = pd.DataFrame({'id1': [1], 'id2': [999]})
    feats = compute_pos_features(pos_nodes, pairs, show_progress=False)
    assert feats[0, :N_TAGS].any()
    assert not feats[0, N_TAGS:].any()


def test_compute_pos_features_row_alignment_and_caching(pos_nodes):
    """Each unique node id is tagged once and reused for every pair it appears in."""
    pairs = pd.DataFrame({'id1': [1, 1, 2], 'id2': [2, 2, 1]}, index=['x', 'y', 'z'])
    feats = compute_pos_features(pos_nodes, pairs, show_progress=False)
    assert feats.shape == (3, 2 * N_TAGS)
    assert feats[0].tolist() == pytest.approx(feats[1].tolist())
    assert feats[2, :N_TAGS].tolist() == pytest.approx(feats[0, N_TAGS:].tolist())


def test_compute_pos_features_self_pair(pos_nodes):
    feats = compute_pos_features(pos_nodes, pd.DataFrame({'id1': [1], 'id2': [1]}),
                                 show_progress=False)
    assert feats[0, :N_TAGS].tolist() == pytest.approx(feats[0, N_TAGS:].tolist())


def test_compute_pos_features_empty_pairs_raises(pos_nodes):
    """`np.vstack([])` on an empty pair set raises; callers must not pass empty
    frames. Pinned so the failure mode stays visible."""
    with pytest.raises(ValueError):
        compute_pos_features(pos_nodes, pd.DataFrame({'id1': [], 'id2': []}),
                             show_progress=False)
