import pandas as pd
import networkx as nx


def load_edges(path: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, index_col='id')


def load_nodes(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load nodes.tsv. Pass nrows for fast sampling during development."""
    return pd.read_csv(path, sep='\t', nrows=nrows, index_col='id')


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    """Build undirected graph from labelled edge pairs (positive edges only)."""
    positive = edges[edges['label'] == 1][['id1', 'id2']]
    G = nx.from_pandas_edgelist(positive, source='id1', target='id2')
    return G
