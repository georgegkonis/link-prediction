# Link Prediction on Text Attributed Graphs

Computer Engineering diploma thesis (University of Patras, CEID). Binary classification of Wikipedia article pairs: does
a hyperlink exist between them?

Datasets: DSAA 2023 Kaggle Competition (Wikipedia subgraph link prediction) and `wiki_cs_8k` (a Computer Science connected subgraph of Simple English Wikipedia crawled from SQL dumps).

**Primary metric:** Macro F1-score. Secondary: AUC-ROC, separate zero-CN and operational cold-start diagnostics, inference latency.

## CascadeLP

Novel four-tier, confidence-based link predictor that routes node pairs through progressively expensive models:

| Tier | Method                     | Feature                                         |
|------|----------------------------|-------------------------------------------------|
| 0    | Self-loop check            | id1 == id2                                      |
| 1    | Structural (LogReg)        | CN, Jaccard, Adamic-Adar, Preferential Attachment |
| 2    | Linguistic (Random Forest) | POS frequency vectors                           |
| 3    | Semantic (LogReg)          | Sentence-Transformer cosine similarity          |

Each tier exits if `max(proba) ≥ threshold`; otherwise the pair cascades to the next tier.

See [docs/design.md](docs/design.md) for architecture details and key decisions.

## Environment

Conda environment: `link-prediction` (Python 3.10, CPU PyTorch).

```bash
conda env create -f environment.yml      # first-time setup
conda env update -f environment.yml --prune  # after changes
```

All `make` targets invoke `conda run -n link-prediction` internally.

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml

# 2. Download dataset (requires Kaggle access)
make run SCRIPT=dsaa.download

# 3. Compute features
make run SCRIPT=dsaa.compute_structural
make run SCRIPT=dsaa.compute_semantic

# 4. Train and evaluate
make train MODEL=cascade
make predict-test MODEL=cascade

# 5. Compile thesis
make latex-compile DOC=thesis
```

The Kaggle notebook still uses the former script paths and needs updating before reuse.
See [docs/commands.md](docs/commands.md#running-on-kaggle).

## Documentation

| Doc                                          | Contents                                     |
|----------------------------------------------|----------------------------------------------|
| [docs/architecture.md](docs/architecture.md) | Repository layout, module reference          |
| [docs/data.md](docs/data.md)                 | Data files, key facts, separability stats    |
| [docs/commands.md](docs/commands.md)         | All make targets and dev flags               |
| [scripts/README.md](scripts/README.md)       | Purpose and output of every script           |
| [docs/design.md](docs/design.md)             | CascadeLP architecture, key design decisions |

## Troubleshooting

- **Kaggle download fails:** Check Kaggle credentials and competition access.
- **Sentence-Transformer takes too long during development:** Use `dev.skip_st=true` with `scripts.dsaa.compute_semantic`.
- **LaTeX build fails:** Run `make latex-clean DOC=thesis` and retry; check `latex/thesis/thesis.log`

## Author

George Gkonis — CEID, University of Patras

## License

Academic use only.
