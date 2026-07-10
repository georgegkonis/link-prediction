# Link Prediction on Wikipedia Graphs

Computer Engineering diploma thesis (University of Patras, CEID). Binary classification of Wikipedia article pairs: does a hyperlink exist between them?

Dataset: DSAA 2023 Kaggle Competition — Wikipedia subgraph link prediction.

**Primary metric:** Macro F1-score. Secondary: AUC-ROC, cold-start F1, inference latency.

## CascadeLP

Novel four-tier, confidence-based link predictor that routes node pairs through progressively expensive models:

| Tier | Method | Feature |
|------|--------|---------|
| 0 | Self-loop check | id1 == id2 |
| 1 | Structural (LogReg) | CN, Jaccard, Adamic-Adar, PageRank |
| 2 | Linguistic (Random Forest) | POS frequency vectors |
| 3 | Semantic (LogReg) | TF-IDF + Sentence-Transformer cosine similarity |

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

# 2. Download dataset (requires KAGGLE_USERNAME / KAGGLE_KEY in .env)
make data-download

# 3. Compute features
make features-structural
make features-semantic

# 4. Train and evaluate
make train MODEL=cascade
make evaluate MODEL=cascade

# 5. Compile thesis
make paper-compile
```

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/architecture.md](docs/architecture.md) | Repository layout, module reference |
| [docs/data.md](docs/data.md) | Data files, key facts, separability stats |
| [docs/commands.md](docs/commands.md) | All make targets and dev flags |
| [docs/design.md](docs/design.md) | CascadeLP architecture, key design decisions |

## Troubleshooting

- **Kaggle download fails:** Check `~/.kaggle/kaggle.json` or set `KAGGLE_USERNAME`/`KAGGLE_KEY` in `.env`
- **Node2Vec is slow:** Use `--skip-n2v` dev flag (see [docs/commands.md](docs/commands.md))
- **Sentence-Transformer hangs:** Use `--skip-st` dev flag
- **LaTeX build fails:** Run `make paper-clean` and retry; check `paper/main.log`

## Author

George Gkonis — CEID, University of Patras

## License

Academic use only.
