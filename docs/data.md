# Data

## Files

| File                                            | Description                                    |
|-------------------------------------------------|------------------------------------------------|
| `data/raw/train.csv`                            | Pairs with labels (id1, id2, label)            |
| `data/raw/test.csv`                             | Unlabeled pairs (id1, id2) for submission      |
| `data/raw/nodes.tsv`                            | Node text (id, markup)                         |
| `data/interim/pp_nodes.csv`                     | Preprocessed node text (id, text)              |
| `data/interim/structural_{train,test}.csv`      | CN, Jaccard, AA, PageRank per pair             |
| `data/interim/node2vec.kv`                      | Trained KeyedVectors (Node2Vec embeddings)     |
| `data/interim/tfidf_{train,test}.csv`           | TF-IDF cosine similarity                       |
| `data/interim/sentence_emb_{train,test}.csv`    | Sentence-Transformer cosine similarity         |
| `data/interim/{pos,tfidf}_train.npy / test.npy` | POS frequency features                         |
| `data/interim/difficulty_{train,test}.csv`      | Per-pair difficulty labels                     |

## Key Facts

Verified via `make analyze-leakage`:

- Train/test pair overlap is negligible (exact and reversed)
- A small number of self-loops appear in both splits; nearly all are positive (one is labeled negative — a label anomaly)
- A small number of intra-train duplicate undirected pairs exist; row-based splits can place the same pair on both sides

## Separability

From `make analyze-dataset`:

- The majority of pairs are genuinely hard (require semantic reasoning)
- A significant minority are trivially resolvable via structural or text-similarity features
- A small fraction are self-loops

Difficulty labels: `trivial_self_loop` / `trivial_high_cn` / `trivial_high_textsim` / `hard`
