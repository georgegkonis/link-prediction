"""Shared synthetic fixtures.

Everything here is built in-code: no `data/raw/`, no `data/interim/`, no network
access, no model downloads. Files that a test genuinely needs are written to
``tmp_path``.
"""

import numpy as np
import networkx as nx
import pandas as pd
import pytest


# ── Graph ────────────────────────────────────────────────────────────────────
#
#   1 --- 2
#   | \  /| \
#   |  \/ |  \
#   |  /\ |   \
#   3 --- 4 --- 5
#
# edges: (1,2) (1,3) (2,3) (2,4) (3,4) (4,5)
# degrees: 1→2, 2→3, 3→3, 4→3, 5→1
TINY_EDGES = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)]


@pytest.fixture
def tiny_graph() -> nx.Graph:
    return nx.Graph(TINY_EDGES)


@pytest.fixture
def tiny_edges_df() -> pd.DataFrame:
    """Edge list in train.csv shape: id index + id1/id2/label."""
    rows = [(u, v, 1) for u, v in TINY_EDGES]
    rows += [(1, 5, 0), (3, 5, 0), (7, 7, 1)]  # negatives + one self-loop
    df = pd.DataFrame(rows, columns=['id1', 'id2', 'label'])
    df.index = pd.RangeIndex(len(df), name='id')
    return df


@pytest.fixture
def tiny_nodes() -> pd.DataFrame:
    """Node text table indexed by id, containing raw wiki markup."""
    texts = {
        1: "{{Infobox city}} '''Paris''' is the capital of [[France]].",
        2: "{{Infobox city}} '''Paris''' is the capital of [[France]].",  # identical to 1
        3: 'Quantum chromodynamics describes the strong interaction.',
        4: 'Quantum chromodynamics describes the strong interaction.',
        5: 'the the the',
    }
    df = pd.DataFrame({'text': pd.Series(texts)})
    df.index.name = 'id'
    return df


# ── Classification toy data ──────────────────────────────────────────────────

@pytest.fixture
def separable_scores() -> tuple[np.ndarray, np.ndarray]:
    """1-D score feature that is linearly separable from its label."""
    rng = np.random.default_rng(0)
    neg = rng.normal(0.1, 0.02, 60)
    pos = rng.normal(0.9, 0.02, 60)
    scores = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)])
    return scores, y


@pytest.fixture
def separable_structural() -> tuple[pd.DataFrame, np.ndarray]:
    """Structural-heuristic frame whose CN column separates the classes."""
    rng = np.random.default_rng(1)
    n = 60
    cn = np.concatenate([rng.integers(0, 2, n), rng.integers(8, 12, n)]).astype(float)
    df = pd.DataFrame({
        'cn': cn,
        'jaccard': cn / 20.0,
        'adamic_adar': cn * 0.9,
        'pref_attach': cn * 3.0,
    })
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return df, y
