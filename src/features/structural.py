import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm


def compute_heuristics(G: nx.Graph, pairs: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """
    Compute Common Neighbours, Jaccard, Adamic-Adar and Preferential Attachment
    for every (id1, id2) pair. Self-loops get NaN; pairs where either node is
    absent from the graph get 0.
    """
    self_loop = pairs['id1'] == pairs['id2']
    non_self = pairs[~self_loop]

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


def train_node2vec(
    G: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 200,
    workers: int = 4,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 10,
) -> object:
    """Train Node2Vec and return the fitted gensim KeyedVectors."""
    n2v = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                   num_walks=num_walks, workers=workers, p=p, q=q, quiet=True)
    model = n2v.fit(window=window, min_count=1, batch_words=4)
    return model.wv


def node2vec_scores(wv, pairs: pd.DataFrame, missing_score: float = 0.0) -> np.ndarray:
    """
    Cosine similarity between Node2Vec embeddings for each pair.
    Returns missing_score when either node has no embedding (cold-start).
    """
    scores = []
    for _, row in pairs.iterrows():
        u, v = str(row['id1']), str(row['id2'])
        if u in wv and v in wv:
            scores.append(float(wv.similarity(u, v)))
        else:
            scores.append(missing_score)
    return np.array(scores)
