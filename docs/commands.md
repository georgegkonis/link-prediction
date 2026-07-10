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
make train MODEL=cascade         # Train CascadeLP — exports per-pair tier/difficulty/correctness
                                 #   to outputs/predictions/cascade_val_tiers.csv
                                 #   Supports: --tier1-threshold, --tier2-threshold,
                                 #             --cn-threshold, --tfidf-threshold
make evaluate MODEL=cascade      # Run inference on test set → outputs/predictions/

make analyze-leakage             # Train/test pair overlap + self-loop + intra-train duplicate audit
make analyze-dataset             # Separability characterization → data/interim/difficulty_{train,test}.csv
make analyze-cascade             # CascadeLP tier and difficulty breakdown

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
