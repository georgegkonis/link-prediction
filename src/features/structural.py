import random

import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
from tqdm import tqdm


def compute_heuristics(G: nx.Graph, pairs: pd.DataFrame, show_progress: bool = True,
                       exclude_target_edge: bool = False) -> pd.DataFrame:
    """
    Compute Common Neighbours, Jaccard, Adamic-Adar and Preferential Attachment
    for every (id1, id2) pair. Self-loops get NaN; pairs where either node is
    absent from the graph get 0.
    """
    if exclude_target_edge:
        rows = []
        for u, v in tqdm(pairs[['id1', 'id2']].itertuples(index=False, name=None),
                         total=len(pairs), disable=not show_progress, desc='Masked heuristics'):
            if u == v:
                rows.append([np.nan] * 4)
                continue
            edge = dict(G[u][v]) if G.has_edge(u, v) else None
            if edge is not None:
                G.remove_edge(u, v)
            try:
                if u not in G or v not in G or G.degree(u) == 0 or G.degree(v) == 0:
                    rows.append([0.0] * 4)
                else:
                    rows.append([len(list(nx.common_neighbors(G, u, v))),
                                 next(nx.jaccard_coefficient(G, [(u, v)]))[2],
                                 next(nx.adamic_adar_index(G, [(u, v)]))[2],
                                 G.degree(u) * G.degree(v)])
            finally:
                if edge is not None:
                    G.add_edge(u, v, **edge)
        return pd.DataFrame(rows, index=pairs.index,
                            columns=['cn', 'jaccard', 'adamic_adar', 'pref_attach'])

    self_loop = pairs['id1'] == pairs['id2']
    non_self = pairs[~self_loop]

    if non_self.empty:
        # `DataFrame.apply(..., axis=1)` on an empty frame yields an empty
        # DataFrame rather than a boolean Series, which breaks the masking below.
        return pd.DataFrame(
            np.nan, index=pairs.index,
            columns=['cn', 'jaccard', 'adamic_adar', 'pref_attach'],
        )

    both_present = non_self.apply(lambda r: G.has_node(r['id1']) and G.has_node(r['id2']), axis=1)
    valid = non_self[both_present]
    cold = non_self[~both_present]

    ebunch = list(zip(valid['id1'], valid['id2']))

    # Common Neighbours
    cn_scores = []
    it = tqdm(ebunch, desc='Common Neighbours', disable=not show_progress)
    for u, v in it:
        cn_scores.append(len(list(nx.common_neighbors(G, u, v))))

    # Bulk networkx generators
    jac = [s for _, _, s in nx.jaccard_coefficient(G, ebunch)]
    aa  = [s for _, _, s in tqdm(nx.adamic_adar_index(G, ebunch),   total=len(ebunch), desc='Adamic-Adar',  disable=not show_progress)]
    pa  = [s for _, _, s in tqdm(nx.preferential_attachment(G, ebunch), total=len(ebunch), desc='Pref-Attach', disable=not show_progress)]

    valid_df = pd.DataFrame(
        {'cn': cn_scores, 'jaccard': jac, 'adamic_adar': aa, 'pref_attach': pa},
        index=valid.index,
    )
    cold_df = pd.DataFrame(
        {'cn': 0.0, 'jaccard': 0.0, 'adamic_adar': 0.0, 'pref_attach': 0.0},
        index=cold.index,
    )

    result = pd.concat([valid_df, cold_df]).reindex(pairs.index)
    return result


def _generate_walks(G: nx.Graph, num_walks: int, walk_length: int, seed: int) -> list[list[str]]:
    """
    Generate uniform random walks over G once, as str node-id tokens.

    gensim's Word2Vec iterates its corpus once to build the vocabulary and
    once per training epoch (5 by default) — a corpus that regenerates walks
    lazily on every pass pays the (expensive, single-threaded) walk-generation
    cost 6x. Materializing the walk list once and reusing it avoids that; at
    this graph's size (~7M walks x 30 tokens) it comfortably fits in memory
    (~2 GB) well inside the 8 GB budget that also has to hold the graph and
    the resulting Word2Vec vocabulary/model.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    to_str = {n: str(n) for n in nodes}
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            cur = start
            for _ in range(walk_length - 1):
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break
                cur = rng.choice(nbrs)
                walk.append(cur)
            walks.append([to_str[n] for n in walk])
    return walks


def train_node2vec(
    G: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 10,
    workers: int = 4,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 10,
    seed: int = 42,
) -> object:
    """
    Train DeepWalk-equivalent node embeddings and return the gensim KeyedVectors.

    With ``p == q == 1`` (the configuration used throughout this thesis) Node2Vec
    reduces to DeepWalk: unbiased, uniform random walks. We generate those walks
    directly and stream them to gensim's Word2Vec, avoiding the eliorc ``node2vec``
    library's O(sum deg^2) transition-probability precompute and its per-worker
    duplication of that table — both of which caused out-of-memory crashes on this
    graph. Only the unbiased case is supported.
    """
    if p != 1.0 or q != 1.0:
        raise NotImplementedError(
            'train_node2vec only supports the unbiased DeepWalk case (p == q == 1); '
            f'got p={p}, q={q}.'
        )
    walks = _generate_walks(G, num_walks=num_walks, walk_length=walk_length, seed=seed)
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window,
        min_count=1,
        sg=1,
        workers=workers,
        seed=seed,
    )
    return model.wv


def node2vec_hadamard_features(wv, pairs: pd.DataFrame, dim: int = 64) -> np.ndarray:
    """
    Element-wise (Hadamard) product of Node2Vec embeddings for each pair
    (Grover & Leskovec, 2016). Rows where either node has no embedding
    (cold-start) are zero vectors.
    """
    key_to_index = wv.key_to_index
    # dtype is pinned so an empty pair set still yields an integer index array
    idx1 = np.array([key_to_index.get(str(u), -1) for u in pairs['id1'].values], dtype=np.int64)
    idx2 = np.array([key_to_index.get(str(v), -1) for v in pairs['id2'].values], dtype=np.int64)
    valid = (idx1 >= 0) & (idx2 >= 0)

    out = np.zeros((len(pairs), dim), dtype=np.float32)
    out[valid] = wv.vectors[idx1[valid]] * wv.vectors[idx2[valid]]
    return out
