# Link Prediction on Wikipedia Graphs

A Computer Engineering diploma thesis (University of Patras, CEID) on link prediction for sparsified Wikipedia
subgraphs. The problem is binary classification of node pairs: does a hyperlink exist between two Wikipedia articles?

## Novel Contribution: CascadeLP

**CascadeLP** is a four-tier, confidence-based link predictor that routes node pairs through progressively expensive
models:

1. **Tier 0 (Self-loops):** Trivial — same node pair
2. **Tier 1 (Structural):** Fast heuristics (Common Neighbors, Jaccard, Adamic-Adar, PageRank)
3. **Tier 2 (Linguistic):** POS frequency features + Random Forest
4. **Tier 3 (Semantic):** TF-IDF + Sentence-Transformer embeddings + Logistic Regression

Each tier outputs a confidence score; if `max(proba) ≥ threshold`, the pair is classified and routed out. Otherwise, it
cascades to the next tier.

The thesis frames CascadeLP primarily as a **cost-aware diagnostic**: measuring which tier resolves which pairs,
cross-referenced against a data-driven difficulty label (self-loop / high-CN / high-textsim / hard), rather than
claiming headline accuracy improvements.

**Primary metric:** Macro F1-score. Secondary: AUC-ROC, cold-start F1, inference latency.

## Environment

**Conda environment:** `link-prediction` (Python 3.10, CPU PyTorch)

```bash
# First-time setup
conda env create -f environment.yml

# After changes
conda env update -f environment.yml --prune
```

All `make` targets invoke `conda run -n link-prediction` internally — no manual activation needed.

## Repository Layout

```
.
├── src/                        Python library (data, features, models, utils)
│   ├── data/                   Data loading and preprocessing
│   ├── features/               Feature engineering (structural, semantic, linguistic)
│   ├── models/                 Classifiers and CascadeLP orchestrator
│   └── utils/                  Evaluation metrics, difficulty labeling, helpers
├── scripts/                    CLI entry points (download, preprocess, train, evaluate, audit_leakage, analyze_dataset)
├── data/
│   ├── raw/                    Source data (.gitignored, 641 MB)
│   └── interim/                Generated features (.gitignored)
├── outputs/
│   ├── checkpoints/            Serialized .joblib models (.gitignored)
│   ├── figures/                PNG plots (.gitignored)
│   └── predictions/            Kaggle submission CSVs (.gitignored)
├── paper/                      LaTeX thesis (CEID bilingual template)
│   ├── main.tex                Entry point
│   ├── body_matter/            chap1–chap6 (Intro, Background, Methodology, Experiments, Analysis, Conclusion)
│   ├── front_matter/           abstract, acknowledgements
│   ├── back_matter/            appendix, references.bib, abbreviations, glossary
│   ├── figures/                PNG plots copied from outputs/figures/
│   └── static/                 hellas.bst, PatrasLogo.png
├── environment.yml
└── Makefile
```

## Common Commands

```bash
make env                         # Create conda environment
make env-update                  # Update from environment.yml
make data-download               # Pull DSAA 2023 dataset from Kaggle (requires .env with KAGGLE_USERNAME/KEY)
make features-structural         # Compute CN/Jaccard/AA/PA/Node2Vec for all pairs
make features-semantic           # Compute TF-IDF, Sentence-Transformer, POS features
make train MODEL=structural      # Train a baseline model (also: tfidf, pos, embedding, svm, cascade)
make train MODEL=cascade         # Train CascadeLP with per-pair tier/difficulty/correctness audit
make evaluate MODEL=cascade      # Run inference on test set → outputs/predictions/
make analyze-leakage             # Train/test pair overlap + self-loop + intra-train duplicate audit
make analyze-dataset             # Separability characterization → data/interim/difficulty_{train,test}.csv
make analyze-cascade             # CascadeLP tier and difficulty breakdown
make paper-compile               # Compile thesis PDF (xelatex → biber → xelatex × 2)
make paper-clean                 # Remove LaTeX auxiliary files (keeps main.pdf)
make jupyter                     # Start JupyterLab
```

### Development Flags

```bash
conda run -n link-prediction python -m scripts.data.compute_structural --nrows 500 --skip-n2v
conda run -n link-prediction python -m scripts.data.compute_semantic   --nrows 300 --skip-st
```

## Data

| File                                            | Rows     | Description                                                                                       |
|-------------------------------------------------|----------|---------------------------------------------------------------------------------------------------|
| `data/raw/train.csv`                            | 948,232  | Pairs with labels (id1, id2, label) — 46% positive, 54% negative                                  |
| `data/raw/test.csv`                             | 238,365  | Unlabeled pairs (id1, id2) for submission                                                         |
| `data/raw/nodes.tsv`                            | 837,855  | Node text (id, markup) — 641 MB                                                                   |
| `data/interim/pp_nodes.csv`                     | 837,855  | Preprocessed node text (id, text)                                                                 |
| `data/interim/structural_{train,test}.csv`      | As above | CN, Jaccard, AA, PageRank per pair                                                                |
| `data/interim/node2vec.kv`                      | —        | Trained KeyedVectors (Node2Vec embeddings)                                                        |
| `data/interim/tfidf_{train,test}.csv`           | As above | TF-IDF cosine similarity                                                                          |
| `data/interim/sentence_emb_{train,test}.csv`    | As above | Sentence-Transformer cosine similarity                                                            |
| `data/interim/{pos,tfidf}_train.npy / test.npy` | As above | POS frequency features (72-dim)                                                                   |
| `data/interim/difficulty_{train,test}.csv`      | As above | Data-driven difficulty labels (trivial_self_loop / trivial_high_cn / trivial_high_textsim / hard) |

**Key Facts** (verified via `make analyze-leakage`):

- 668K nodes in structural graph
- 10,221 self-loops in train (10,220 positive, 1 negative — label anomaly)
- 2,553 self-loops in test
- Exact train/test pair overlap: 2 pairs (negligible)
- Reversed-pair overlap: 301 pairs (negligible)
- Intra-train duplicate pairs: 642 undirected groups

**Separability** (from `make analyze-dataset`):

- 64% of pairs are genuinely hard (require semantic reasoning)
- 35% are trivial (resolvable via structural or text-similarity features)
- 1% are self-loops

## Architecture: src/

### Data Layer

| Module                      | Purpose                                                                  |
|-----------------------------|--------------------------------------------------------------------------|
| `src/data/loader.py`        | Stream 641MB nodes.tsv safely; 50K-row chunks via `load_nodes_for_ids()` |
| `src/data/infobox.py`       | Parse `{{infobox}}` wiki markup → `Infobox(title, content, other)`       |
| `src/data/description.py`   | Extract article prose → `Description(text, keywords)`                    |
| `src/data/preprocessing.py` | Text cleaning (spaces, HTML, infobox extraction, merging)                |

### Features

| Module                       | Purpose                                                               |
|------------------------------|-----------------------------------------------------------------------|
| `src/features/structural.py` | Graph heuristics (CN, Jaccard, AA, PA); Node2Vec training and scoring |
| `src/features/embeddings.py` | TF-IDF and Sentence-Transformer feature computation                   |
| `src/features/linguistic.py` | POS frequency vectors (36-dim per node, 72-dim concatenated)          |

### Models

| Module                  | Purpose                                                                                                              |
|-------------------------|----------------------------------------------------------------------------------------------------------------------|
| `src/models/svm.py`     | Five sklearn classifiers (LogReg + RBF SVC) with shared `.fit()`, `.predict_proba()`, `.save()`, `.load()` interface |
| `src/models/cascade.py` | `CascadeLP` orchestrator — routes pairs through tiers based on confidence thresholds                                 |
| `src/models/gnn.py`     | Stub (Phase 5) — GNN with MC-Dropout uncertainty                                                                     |

### Evaluation & Difficulty Labeling

| Symbol                                        | Purpose                                                                                   |
|-----------------------------------------------|-------------------------------------------------------------------------------------------|
| `src/utils/metrics.EvalResult`                | Dataclass: macro_f1, auc_roc, cold_start_f1, latency_ms                                   |
| `src/utils/metrics.cold_start_mask(pairs, G)` | Boolean mask for pairs with zero common neighbors                                         |
| `src/utils/metrics.evaluate()`                | Full eval including cold-start subset                                                     |
| `src/utils/metrics.evaluate_by_group()`       | Per-group stats (by tier, difficulty, etc.)                                               |
| `src/utils/difficulty.label_difficulty()`     | Assign per-pair labels: trivial_self_loop / trivial_high_cn / trivial_high_textsim / hard |
| `src/utils/difficulty.pick_thresholds()`      | Data-driven thresholds (1% FPR on negative class)                                         |

## Papers & References

- **Thesis:** `/paper/main.tex` — CEID bilingual template (Greek/English)
- **BibTeX:** `paper/back_matter/references.bib`

## Key Design Decisions

1. **LogisticRegression over SVC** in core tiers: SVC RBF is O(n²–n³); infeasible on 948K pairs. LogReg trains in
   seconds. Real `SvmClassifier` exists as a separate baseline (RBF SVC on 20K stratified subsample).

2. **POS at Tier 2:** Reproduces DSAA 2023 baseline (F1=0.99999) and is fast at inference (no transformer).

3. **Stream nodes.tsv in 50K-row chunks:** File is 641MB; keep only needed IDs in memory.

4. **Filter self-loops from training:** 10,220 of 10,221 are positive; would inflate metrics. Handled separately in Tier
    0.

5. **Three independent tiers:** Each trained on full training set; no cascading labels between tiers.

6. **Copy figures to paper/figures/:** Figures in `outputs/figures/` are gitignored (ephemeral). Copying to
   `paper/figures/` (tracked in git) ensures thesis is self-contained and reproducible at any commit.

## No Test Suite

Validation is done via stratified train/val splits in `scripts/train.py`. See `Makefile` dev flags for iteration.

## Getting Started

1. Clone and activate environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Download dataset (requires Kaggle credentials):
   ```bash
   make data-download
   ```

3. Preprocess and generate features:
   ```bash
   make preprocess
   make features-structural
   make features-semantic
   ```

4. Train and evaluate:
   ```bash
   make train MODEL=cascade
   make evaluate MODEL=cascade
   make analyze-dataset
   ```

5. Compile thesis:
   ```bash
   make paper-compile
   ```

## Troubleshooting

- **Kaggle download fails:** Check `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY` in `.env`)
- **Node2Vec is slow:** Use `--skip-n2v` dev flag; see [Makefile](Makefile)
- **Sentence-Transformer download hangs:** Use `--skip-st` dev flag
- **LaTeX build fails:** Run `make paper-clean` and retry; check `paper/main.log` for errors

## Author

George Gkonis (CEID, University of Patras)

## License

Thesis and code are for academic use only.
