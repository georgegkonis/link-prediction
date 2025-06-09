import pandas as pd
import networkx as nx


def load_edges(path: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, index_col='id')


def load_nodes(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load nodes.tsv. Pass nrows for fast sampling during development."""
    return pd.read_csv(path, sep='\t', nrows=nrows, index_col='id')


def load_nodes_for_ids(path: str, node_ids: set[int], chunksize: int = 50_000) -> pd.DataFrame:
    """
    Stream nodes.tsv in chunks and keep only the rows whose ID is in node_ids.
    Much more memory-efficient than loading the full 641MB file when only a
    subset of nodes is needed.
    """
    chunks = []
    for chunk in pd.read_csv(path, sep='\t', index_col='id', chunksize=chunksize):
        needed = chunk[chunk.index.isin(node_ids)]
        if not needed.empty:
            chunks.append(needed)
    return pd.concat(chunks) if chunks else pd.DataFrame(columns=['text'])


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    """Build undirected graph from labelled edge pairs (positive edges only)."""
    positive = edges[edges['label'] == 1][['id1', 'id2']]
    G = nx.from_pandas_edgelist(positive, source='id1', target='id2')
    return G
