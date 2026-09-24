# Data

We use two datasets:
1. **DSAA 2023 Kaggle Dataset**: The original competition dataset (Wikipedia subgraph link prediction).
2. **Wiki-CS-8k Dataset**: A Computer Science connected subgraph of Simple English Wikipedia crawled from SQL dumps.

The data pipelines strictly partition files into `dsaa/` and `wiki_cs_8k/` subdirectories.

The generated `wiki_cs_8k_sparse20/` benchmark references the same 8,000
articles and complete verified edge universe. Its `benchmark.json` records the
20% observed-edge sample, source/file hashes, and the train, random-test, and
hard-test pair files. The two test suites share all withheld positive links but
use disjoint negative pairs.

## Files

The table below describes the files for a given dataset (substitute `{dataset}` with `dsaa` or `wiki_cs_8k`):

| File                                            | Description                                    |
|-------------------------------------------------|------------------------------------------------|
| `data/raw/{dataset}/train.csv`                       | Pairs with labels (id1, id2, label)            |
| `data/raw/{dataset}/test.csv`                        | Unlabeled pairs (id1, id2) for submission      |
| `data/raw/{dataset}/nodes.tsv`                       | Node text (id, markup)                         |
| `data/interim/{dataset}/pp_nodes.csv`                | Preprocessed node text (id, text)              |
| `data/interim/{dataset}/structural_{train,test}.csv` | CN, Jaccard, AA, Preferential Attachment per pair |
| `data/interim/{dataset}/tfidf_{train,test}.csv`      | TF-IDF cosine similarity                       |
| `data/interim/{dataset}/sentence_emb_{train,test}.csv`| Sentence-Transformer cosine similarity         |
| `data/interim/{dataset}/{pos,tfidf}_train.npy / test.npy` | POS frequency features                         |
| `data/interim/{dataset}/difficulty_{train,test}.csv` | Per-pair difficulty labels                     |

## Key Facts

Verified via `make run SCRIPT=dsaa.audit_pairs`:

- Train/test pair overlap is negligible (exact and reversed)
- A small number of self-loops appear in both splits; nearly all are positive (one is labeled negative — a label anomaly)
- A small number of intra-train duplicate undirected pairs exist; row-based splits can place the same pair on both sides

## Separability

From `make run SCRIPT=dsaa.label_difficulty`:

- The majority of pairs fail the selected simple separability criteria; this does not prove that they require semantic reasoning
- A significant minority are trivially resolvable via structural or text-similarity features
- A small fraction are self-loops

Difficulty labels: `trivial_self_loop` / `trivial_high_cn` / `trivial_high_textsim` / `hard`
