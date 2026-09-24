import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.embeddings import clean_wiki_text

# All universal POS tags produced by the averaged_perceptron_tagger
_POS_TAGS = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
    'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
    'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
    'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
    'WP', 'WP$', 'WRB',
]
_TAG_INDEX = {tag: i for i, tag in enumerate(_POS_TAGS)}


def ensure_nltk_data():
    for resource in ('punkt_tab', 'averaged_perceptron_tagger_eng'):
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'taggers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


def pos_frequency_vector(text: str) -> np.ndarray:
    """
    Return a normalised frequency vector over _POS_TAGS for the given text.
    Vector sums to 1 (or is all-zeros for empty/unparseable input).
    """
    ensure_nltk_data()
    clean = clean_wiki_text(text)
    if not clean:
        return np.zeros(len(_POS_TAGS))

    tokens = nltk.word_tokenize(clean)
    if not tokens:
        return np.zeros(len(_POS_TAGS))

    tags   = nltk.pos_tag(tokens)
    counts = Counter(tag for _, tag in tags)
    total  = sum(counts.values()) or 1

    vec = np.zeros(len(_POS_TAGS))
    for tag, count in counts.items():
        if tag in _TAG_INDEX:
            vec[_TAG_INDEX[tag]] = count / total
    return vec


def compute_pos_features(
    nodes: pd.DataFrame,
    pairs: pd.DataFrame,
    show_progress: bool = True,
) -> np.ndarray:
    """
    For each pair return the concatenation of the POS frequency vectors of
    id1 and id2.  Shape: (n_pairs, 2 * n_pos_tags).
    Missing nodes get a zero vector.
    """
    unique_ids = pd.unique(pairs[['id1', 'id2']].values.ravel())
    node_vecs = encode_pos_nodes(nodes, unique_ids, show_progress=show_progress)
    return assemble_pos_features(node_vecs, pairs)


def encode_pos_nodes(
    nodes: pd.DataFrame,
    node_ids,
    show_progress: bool = True,
) -> dict[int, np.ndarray]:
    """Compute one reusable POS-frequency vector per requested article."""
    ensure_nltk_data()
    node_vecs: dict[int, np.ndarray] = {}

    for node_id in tqdm(node_ids, desc='POS tagging', disable=not show_progress):
        node_id = int(node_id)
        if node_id in nodes.index:
            node_vecs[node_id] = pos_frequency_vector(nodes.loc[node_id, 'text'])
        else:
            node_vecs[node_id] = np.zeros(len(_POS_TAGS))
    return node_vecs


def assemble_pos_features(
    node_vecs: dict[int, np.ndarray],
    pairs: pd.DataFrame,
) -> np.ndarray:
    """Concatenate cached endpoint vectors for a batch of pairs."""
    if pairs.empty:
        return np.empty((0, 2 * len(_POS_TAGS)))

    zero = np.zeros(len(_POS_TAGS))
    return np.vstack([
        np.concatenate([node_vecs.get(int(u), zero), node_vecs.get(int(v), zero)])
        for u, v in pairs[['id1', 'id2']].itertuples(index=False, name=None)
    ])
