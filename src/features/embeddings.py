import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_wiki_text(text: str) -> str:
    """Strip the most common Wikipedia markup so embeddings see clean prose."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)   # HTML comments
    text = re.sub(r'<ref[^>]*>.*?</ref>', ' ', text, flags=re.DOTALL)  # <ref>
    text = re.sub(r'<[^>]+>', ' ', text)                        # HTML tags
    text = re.sub(r'{{[^}]*}}', ' ', text)                      # {{templates}}
    text = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', text) # [[Link|text]] → text
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)  # [url text] → text
    text = re.sub(r"'{2,}", '', text)                            # bold/italic markers
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── TF-IDF ───────────────────────────────────────────────────────────────────

def build_tfidf(texts: list[str], max_features: int = 50_000) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True, min_df=2)
    vectorizer.fit(texts)
    return vectorizer


def compute_tfidf_scores(
    vectorizer: TfidfVectorizer,
    nodes: pd.DataFrame,
    pairs: pd.DataFrame,
) -> np.ndarray:
    """
    Cosine similarity between TF-IDF vectors of id1 and id2 for each pair.
    Returns 0.0 when either node has no text.
    """
    unique_ids = pd.unique(pairs[['id1', 'id2']].values.ravel())
    present = [i for i in unique_ids if i in nodes.index]

    texts  = [clean_wiki_text(nodes.loc[i, 'text']) for i in present]
    matrix = vectorizer.transform(texts)
    id_to_row = {node_id: idx for idx, node_id in enumerate(present)}

    scores = []
    for _, row in pairs.iterrows():
        u, v = row['id1'], row['id2']
        if u in id_to_row and v in id_to_row:
            sim = cosine_similarity(matrix[id_to_row[u]], matrix[id_to_row[v]])[0, 0]
            scores.append(float(sim))
        else:
            scores.append(0.0)
    return np.array(scores)


# ── Sentence Transformers ────────────────────────────────────────────────────

def encode_nodes(
    nodes: pd.DataFrame,
    node_ids: list[int],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 64,
    show_progress: bool = True,
) -> dict[int, np.ndarray]:
    """
    Encode a set of node texts with a sentence transformer.
    Returns {node_id: embedding_vector}.
    Only encodes node_ids that exist in nodes.index.
    """
    model = SentenceTransformer(model_name)
    present = [i for i in node_ids if i in nodes.index]
    texts   = [clean_wiki_text(nodes.loc[i, 'text']) for i in present]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit vectors → dot product = cosine similarity
    )
    return dict(zip(present, embeddings))


def compute_embedding_scores(
    embeddings: dict[int, np.ndarray],
    pairs: pd.DataFrame,
    missing_score: float = 0.0,
) -> np.ndarray:
    """Cosine similarity (dot product of unit vectors) for each pair."""
    scores = []
    for _, row in pairs.iterrows():
        u, v = row['id1'], row['id2']
        if u in embeddings and v in embeddings:
            scores.append(float(np.dot(embeddings[u], embeddings[v])))
        else:
            scores.append(missing_score)
    return np.array(scores)
