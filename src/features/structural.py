import random

import numpy as np
import pandas as pd
import networkx as nx
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


