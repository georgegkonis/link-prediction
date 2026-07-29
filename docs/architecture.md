# Architecture

## Repository Layout

```
.
├── src/                        Python library (data, features, models, utils)
│   ├── data/                   Data loading and preprocessing
│   ├── features/               Feature engineering (structural, semantic, linguistic)
│   ├── models/                 Classifiers and CascadeLP orchestrator
│   └── utils/                  Evaluation metrics, difficulty labeling, helpers
├── scripts/                    CLI entry points
│   ├── data/                   download_data, compute_structural, compute_semantic
│   ├── analysis/               audit_leakage, analyze_dataset, analyze_cascade, ablations
│   ├── paper/                  generate_macros, generate_figures
│   └── (root)                  train, evaluate, submit_kaggle
├── data/
│   ├── raw/                    Source data
│   └── interim/                Generated features
├── outputs/
│   ├── checkpoints/            Serialized .joblib models
│   ├── figures/                PNG plots
│   └── predictions/            Kaggle submission CSVs
└── paper/                      LaTeX thesis (CEID bilingual template)
    ├── main.tex                Entry point
    ├── body_matter/            chap1–chap6
    ├── front_matter/           abstract, acknowledgements
    ├── back_matter/            appendix, references.bib, abbreviations, glossary
    ├── figures/                Plots included in the paper
    └── static/                 PatrasLogo.png
```

## Data Layer (`src/data/`)

| File               | Purpose                         | Key symbols                                                                                 |
|--------------------|---------------------------------|---------------------------------------------------------------------------------------------|
| `loader.py`        | Stream nodes.tsv in chunks      | `load_edges()`, `load_nodes_for_ids()` (chunked), `build_graph()`                           |
| `infobox.py`       | Parse `{{infobox}}` wiki markup | `Infobox` — `title`, `content` dict, `other` list                                           |
| `description.py`   | Extract article prose           | `Description` — `text`, bold/italic `keywords`                                              |
| `preprocessing.py` | Text cleaning pipeline          | `remove_extra_spaces()`, `remove_html_comments()`, `extract_infobox()`, `merge_infoboxes()` |

## Features (`src/features/`)

| File            | Purpose                   | Key symbols                                                                                                                     |
|-----------------|---------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `structural.py` | Graph topology heuristics | `compute_heuristics(G, pairs)` → CN/Jaccard/AA/PA DataFrame; `train_node2vec()`; `node2vec_scores()`                            |
| `embeddings.py` | Text similarity features  | `clean_wiki_text()`; `build_tfidf()`; `compute_tfidf_scores()`; `encode_nodes()` (single-pass ST); `compute_embedding_scores()` |
| `linguistic.py` | NLP features              | `pos_frequency_vector()` → per-node POS vector; `compute_pos_features()` → concatenated (id1 ∥ id2)                             |

## Models (`src/models/`)

| File         | Purpose                      | Key symbols                                                                                                                                                                      |
|--------------|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `svm.py`     | Five sklearn classifiers     | `StructuralClassifier` (LogReg), `TfidfClassifier` (LogReg), `PosClassifier` (RandomForest), `EmbeddingClassifier` (LogReg), `SvmClassifier` (RBF SVC on a stratified subsample) |
| `cascade.py` | Novel four-tier orchestrator | `CascadeLP(tier1_threshold, tier2_threshold)` — `.fit()`, `.predict()` → `(predictions, tier_used)`, `.tier_stats()`                                                             |
| `gnn.py`     | Stub (Phase 5)               | GNN with MC-Dropout uncertainty (not implemented)                                                                                                                                |

## Evaluation & Difficulty Labeling (`src/utils/`)

| Symbol                                                                     | Purpose                                                                           |
|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `metrics.EvalResult`                                                       | Dataclass: macro_f1, auc_roc, cold_start_f1, latency_ms                           |
| `metrics.cold_start_mask(pairs, G)`                                        | Boolean mask — True for pairs with zero common neighbours                         |
| `metrics.evaluate(y_true, y_pred, y_scores, cs_mask)`                      | Full eval including cold-start subset                                             |
| `metrics.evaluate_by_group(y_true, y_pred, group)`                         | Per-group n/accuracy/macro_f1 (e.g. by tier or difficulty label)                  |
| `metrics.tier_difficulty_breakdown(y_true, y_pred, tier_used, difficulty)` | Cross-tab of (tier, difficulty) → n/accuracy/macro_f1                             |
| `metrics.timer()`                                                          | Context manager returning elapsed ms                                              |
| `difficulty.label_difficulty(pairs, cn, tfidf, cn_thr, tfidf_thr)`         | Per-pair label: trivial_self_loop / trivial_high_cn / trivial_high_textsim / hard |
| `difficulty.pick_thresholds(y, cn, tfidf, fpr)`                            | Data-driven trivial-pair thresholds (percentile of negative-class distribution)   |
