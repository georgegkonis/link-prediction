# Commands

All `make` targets invoke `conda run -n link-prediction` internally — no manual activation needed.

## Make Targets

```bash
make env                         # Create conda environment
make env-update                  # Update from environment.yml

make data-download               # Pull DSAA 2023 dataset from Kaggle (requires .env with KAGGLE_USERNAME/KEY)

make features-structural         # Compute CN/Jaccard/AA/PA/Node2Vec for all pairs
make features-semantic           # Compute TF-IDF, Sentence-Transformer, POS features

make train MODEL=structural      # Train a baseline model (also: tfidf, pos, embedding, svm)
make train MODEL=cascade         # Train CascadeLP — exports per-pair tier/difficulty/correctness/score
                                 #   to outputs/predictions/cascade_val_tiers.csv
                                 #   Supports: model.tier1_threshold=, model.tier2_threshold=,
                                 #             training.cn_threshold=, training.tfidf_threshold=,
                                 #             training.no_n2v=true   (thesis-reported heuristics-only
                                 #             config — pass this to reproduce cascade.joblib as documented)
make evaluate MODEL=cascade      # Run inference on test set → outputs/predictions/
                                 #   Pass the same training.no_n2v= value used at train time, or
                                 #   predict() will crash on a feature-dimension mismatch.

make analyze-leakage             # Train/test pair overlap + self-loop + intra-train duplicate audit
make analyze-dataset             # Separability characterization → data/interim/difficulty_{train,test}.csv
make analyze-cascade             # CascadeLP tier and difficulty breakdown
make analyze-node2vec            # Node2Vec with/without ablation + test-set vocabulary coverage
                                 #   → outputs/predictions/node2vec_ablation.json
make analyze-hard-residual       # Tier-3 hard-residual / nodes.tsv join (missing-text analysis)
                                 #   → outputs/predictions/hard_residual_analysis.json
make benchmark-throughput        # CPU inference throughput benchmark
                                 #   → outputs/predictions/throughput_benchmark.json
make ablate-thresholds           # tau1/tau2 threshold grid sweep on the trained cascade checkpoint
                                 #   → outputs/predictions/cascade_threshold_ablation.csv
make analyze                     # Run analyze-leakage, analyze-dataset, analyze-cascade,
                                 #   analyze-node2vec, analyze-hard-residual, benchmark-throughput

make paper-compile               # Compile thesis PDF (xelatex → biber → xelatex × 2)
make paper-clean                 # Remove LaTeX auxiliary files (keeps main.pdf)

make jupyter                     # Start JupyterLab
```

## Dev Flags

For fast iteration without running full feature computation:

```bash
# Skip Node2Vec (slow graph embedding training)
conda run -n link-prediction python -m scripts.data.compute_structural --nrows 500 --skip-n2v

# Skip Sentence-Transformer (slow model download + inference)
conda run -n link-prediction python -m scripts.data.compute_semantic --nrows 300 --skip-st
```
