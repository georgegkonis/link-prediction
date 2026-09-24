import numpy as np
import pandas as pd
import pytest

from src.features import embeddings as emb
from src.features.embeddings import (
    build_tfidf,
    clean_wiki_text,
    compute_cached_tfidf_scores,
    compute_embedding_scores,
    compute_tfidf_scores,
    encode_nodes,
    encode_tfidf_nodes,
)


# ── clean_wiki_text ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('raw, expected', [
    ('{{Infobox city|pop=1}} Hello', 'Hello'),
    ('[[Paris|the city]] is nice', 'the city is nice'),
    ('[[Paris]] is nice', 'Paris is nice'),
    ('<ref name="a">Smith 1999</ref>Text', 'Text'),
    ("'''bold''' and ''italic''", 'bold and italic'),
    ('<!-- hidden note -->visible', 'visible'),
    ('[http://example.com Example] link', 'Example link'),
    ('<div class="x">body</div>', 'body'),
    ('a\n\n  b\t c', 'a b c'),
    ('   padded   ', 'padded'),
    ('', ''),
])
def test_clean_wiki_text_cases(raw, expected):
    assert clean_wiki_text(raw) == expected


@pytest.mark.parametrize('bad', [None, float('nan'), 42, [1, 2]])
def test_clean_wiki_text_non_string_returns_empty(bad):
    assert clean_wiki_text(bad) == ''


def test_clean_wiki_text_does_not_strip_section_headings():
    """Known gap: `== Heading ==` markup survives cleaning. Pinned so a future
    change to the cleaner is a deliberate decision, not an accident."""
    assert clean_wiki_text('== History ==\nText') == '== History == Text'


def test_clean_wiki_text_nested_templates_leak_markup():
    """Known limitation: the template regex `{{[^}]*}}` is non-recursive, so a
    nested template leaves the outer closing braces behind."""
    assert clean_wiki_text('{{a {{b}} c}} tail') == 'c}} tail'


def test_clean_wiki_text_is_idempotent_on_clean_prose():
    prose = 'Paris is the capital of France.'
    assert clean_wiki_text(clean_wiki_text(prose)) == prose


# ── build_tfidf / compute_tfidf_scores ───────────────────────────────────────

CORPUS = [
    'apple banana fruit',
    'apple banana fruit',
    'cherry date sugar',
    'cherry date sugar',
]


def test_build_tfidf_respects_min_df_two():
    vec = build_tfidf(CORPUS)
    assert set(vec.vocabulary_) == {'apple', 'banana', 'fruit', 'cherry', 'date', 'sugar'}
    # a term appearing in only one document is dropped by min_df=2
    vec2 = build_tfidf(CORPUS + ['zebra apple banana fruit'])
    assert 'zebra' not in vec2.vocabulary_


def test_build_tfidf_max_features_cap():
    vec = build_tfidf(CORPUS, max_features=2)
    assert len(vec.vocabulary_) == 2


@pytest.fixture
def tfidf_nodes() -> pd.DataFrame:
    df = pd.DataFrame({'text': {
        1: "{{Infobox}} '''apple''' banana fruit",
        2: "'''apple''' banana fruit",         # same content words as node 1
        3: 'cherry date sugar',
        4: 'apple cherry',
    }})
    df.index.name = 'id'
    return df


def test_compute_tfidf_scores_identical_and_disjoint_text(tfidf_nodes):
    vec = build_tfidf(CORPUS)
    pairs = pd.DataFrame(
        {'id1': [1, 1, 1, 1], 'id2': [2, 3, 1, 999]},
        index=[10, 11, 12, 13],
    )
    scores = compute_tfidf_scores(vec, tfidf_nodes, pairs)
    assert scores.shape == (4,)
    assert scores[0] == pytest.approx(1.0)      # identical content after cleaning
    assert scores[1] == pytest.approx(0.0)      # disjoint vocabulary
    assert scores[2] == pytest.approx(1.0)      # a node against itself
    assert scores[3] == pytest.approx(0.0)      # node 999 absent from `nodes`


def test_compute_tfidf_scores_partial_overlap_is_strictly_between(tfidf_nodes):
    vec = build_tfidf(CORPUS)
    pairs = pd.DataFrame({'id1': [1, 3], 'id2': [4, 4]})
    scores = compute_tfidf_scores(vec, tfidf_nodes, pairs)
    assert 0.0 < scores[0] < 1.0                # share 'apple'
    assert 0.0 < scores[1] < 1.0                # share 'cherry'


def test_compute_tfidf_scores_all_nodes_missing_returns_zeros(tfidf_nodes):
    vec = build_tfidf(CORPUS)
    pairs = pd.DataFrame({'id1': [900, 901], 'id2': [902, 903]})
    scores = compute_tfidf_scores(vec, tfidf_nodes, pairs)
    assert scores.tolist() == [0.0, 0.0]


def test_compute_tfidf_scores_out_of_vocabulary_text_scores_zero():
    vec = build_tfidf(CORPUS)
    nodes = pd.DataFrame({'text': {1: 'zebra quokka', 2: 'zebra quokka'}})
    scores = compute_tfidf_scores(vec, nodes, pd.DataFrame({'id1': [1], 'id2': [2]}))
    # both rows are all-zero vectors → cosine is 0, not NaN
    assert scores[0] == pytest.approx(0.0)
    assert not np.isnan(scores[0])


def test_compute_tfidf_scores_cleans_markup_before_vectorising():
    """Node 1 differs from node 2 only in wiki markup; after cleaning they must
    be identical."""
    vec = build_tfidf(CORPUS)
    nodes = pd.DataFrame({'text': {
        1: "{{Infobox|x=1}} [[Apple|apple]] '''banana''' fruit <ref>x</ref>",
        2: 'apple banana fruit',
    }})
    scores = compute_tfidf_scores(vec, nodes, pd.DataFrame({'id1': [1], 'id2': [2]}))
    assert scores[0] == pytest.approx(1.0)


def test_tfidf_node_cache_reuses_transformed_articles_across_batches(tfidf_nodes):
    vec = build_tfidf(CORPUS)
    cache = encode_tfidf_nodes(vec, tfidf_nodes, [1, 2, 3, 999])
    first = compute_cached_tfidf_scores(
        cache, pd.DataFrame({'id1': [1], 'id2': [2]}))
    second = compute_cached_tfidf_scores(
        cache, pd.DataFrame({'id1': [1, 999], 'id2': [3, 2]}))

    assert first.tolist() == pytest.approx([1.0])
    assert second.tolist() == pytest.approx([0.0, 0.0])


# ── compute_embedding_scores ─────────────────────────────────────────────────

def test_compute_embedding_scores_cosine_of_unit_vectors():
    e = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 1.0]),
        3: np.array([1.0, 0.0]),
        4: np.array([-1.0, 0.0]),
    }
    pairs = pd.DataFrame({'id1': [1, 1, 1, 1], 'id2': [3, 2, 4, 1]})
    scores = compute_embedding_scores(e, pairs)
    assert scores.tolist() == pytest.approx([1.0, 0.0, -1.0, 1.0])


def test_compute_embedding_scores_missing_node_uses_missing_score():
    e = {1: np.array([1.0, 0.0])}
    pairs = pd.DataFrame({'id1': [1, 99], 'id2': [99, 98]})
    assert compute_embedding_scores(e, pairs).tolist() == [0.0, 0.0]
    assert compute_embedding_scores(e, pairs, missing_score=-1.0).tolist() == [-1.0, -1.0]


def test_compute_embedding_scores_row_alignment_with_custom_index():
    e = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    pairs = pd.DataFrame({'id1': [1, 1], 'id2': [1, 2]}, index=[77, 88])
    assert compute_embedding_scores(e, pairs).tolist() == pytest.approx([1.0, 0.0])


def test_compute_embedding_scores_empty_pairs():
    out = compute_embedding_scores({}, pd.DataFrame({'id1': [], 'id2': []}))
    assert out.shape == (0,)


# ── encode_nodes (fake transformer injected — no model download) ─────────────

class _FakeSentenceTransformer:
    """Deterministic stand-in for SentenceTransformer.

    Records the texts it was handed so the test can assert the cleaning step ran.
    """

    seen_texts: list[str] = []
    seen_kwargs: dict = {}
    init_args: dict = {}

    def __init__(self, model_name, device=None):
        type(self).init_args = {'model_name': model_name, 'device': device}

    def encode(self, texts, **kwargs):
        type(self).seen_texts = list(texts)
        type(self).seen_kwargs = kwargs
        # unit vector per text, deterministic in its length
        out = np.zeros((len(texts), 2), dtype=np.float32)
        for i, t in enumerate(texts):
            angle = (len(t) % 7) / 7.0 * np.pi / 2
            out[i] = (np.cos(angle), np.sin(angle))
        return out


@pytest.fixture
def fake_st(monkeypatch):
    _FakeSentenceTransformer.seen_texts = []
    monkeypatch.setattr(emb, 'SentenceTransformer', _FakeSentenceTransformer)
    return _FakeSentenceTransformer


def test_encode_nodes_only_encodes_present_ids(fake_st, tfidf_nodes):
    out = encode_nodes(tfidf_nodes, [1, 3, 999], show_progress=False)
    assert set(out) == {1, 3}
    assert len(fake_st.seen_texts) == 2


def test_encode_nodes_passes_cleaned_text(fake_st, tfidf_nodes):
    encode_nodes(tfidf_nodes, [1], show_progress=False)
    assert fake_st.seen_texts == ['apple banana fruit']


def test_encode_nodes_requests_normalised_embeddings(fake_st, tfidf_nodes):
    encode_nodes(tfidf_nodes, [1, 2], batch_size=8, show_progress=False)
    assert fake_st.seen_kwargs['normalize_embeddings'] is True
    assert fake_st.seen_kwargs['convert_to_numpy'] is True
    assert fake_st.seen_kwargs['batch_size'] == 8
    assert fake_st.seen_kwargs['show_progress_bar'] is False


def test_encode_nodes_output_feeds_compute_embedding_scores(fake_st, tfidf_nodes):
    out = encode_nodes(tfidf_nodes, [1, 2, 3], show_progress=False)
    pairs = pd.DataFrame({'id1': [1, 1], 'id2': [2, 3]})
    scores = compute_embedding_scores(out, pairs)
    assert scores.shape == (2,)
    assert np.all(np.abs(scores) <= 1.0 + 1e-6)
    # nodes 1 and 2 clean to the same text → identical vectors → cosine 1
    assert scores[0] == pytest.approx(1.0, abs=1e-6)


def test_encode_nodes_empty_selection_returns_empty_dict(fake_st, tfidf_nodes):
    assert encode_nodes(tfidf_nodes, [999], show_progress=False) == {}
