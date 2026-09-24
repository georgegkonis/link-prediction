# Commands

All `make` targets invoke `conda run -n link-prediction` internally — no manual activation needed.

## Make Targets

```bash
make env                               # Create conda environment
make env-update                        # Update from environment.yml

# Run an individual entry point
make run SCRIPT=dsaa.download           # Pull DSAA 2023 dataset from Kaggle
make run SCRIPT=dsaa.compute_structural # Compute original-protocol graph features
make run SCRIPT=dsaa.compute_semantic   # Compute text features
make run SCRIPT=dsaa.label_difficulty   # Label difficulty categories
make run SCRIPT=thesis.generate_figures # Generate thesis figures from summary_stats.json

# Dedicated Training Targets
make train MODEL=structural            # Train a baseline model
make train MODEL=cascade               # Train CascadeLP
make predict-test MODEL=cascade         # Write predictions for unlabelled DSAA test pairs
make dsaa-train-all                     # Train all six DSAA models
make compare-matched                   # Equal-size DSAA comparison (3 x 20k samples)
make kaggle-submit FILE=path.csv MSG=run WAIT=--wait # Optional score polling
make kaggle-check WAIT=--wait          # Refresh competition scores

# Rebuild local experiments and thesis artifacts
make pipeline-dsaa                     # Needs Kaggle data access; does not submit predictions
make pipeline-wiki                     # Needs MediaWiki SQL dumps in /tmp and Hugging Face access
make pipeline-wiki-sparse              # Sparse graph; random-negative training
make pipeline-wiki-sparse-mixed        # Same test suites; mixed-negative training
make build-thesis                      # Needs completed experiment outputs and Kaggle score log

# Compilation
make latex-compile DOC=thesis          # Compile thesis PDF
make latex-clean DOC=thesis            # Clean LaTeX auxiliary files
make latex-compile DOC=presentation    # Compile presentation PDF

make jupyter                           # Start JupyterLab
```

## Dev Flags

For fast iteration without running full feature computation:

```bash

# Skip Sentence-Transformer (slow model download + inference)
conda run -n link-prediction python -m scripts.dsaa.compute_semantic dev.nrows=300 dev.skip_st=true
```

Development runs save features under `data/interim/dsaa/dev_.../`; for example,
the command above writes to `dev_nrows_300_skip_st/`. They do not replace the
full-data caches. Both DSAA feature commands honor
`paths.raw` and `paths.interim` Hydra overrides. Model training and prediction
save their resolved configurations beside the results; use matching `tag=NAME`
in both commands for a tagged checkpoint.
The CascadeLP protocol audit, threshold sweep, and inference benchmark read
the untagged `cascade_run_config.json` when available, so their validation
split matches the trained model. Use each command's path flags for results
outside the default output directory.

## Wiki-CS-8k Pipeline

For Wiki-CS-8k, run these steps in order after placing the MediaWiki SQL dumps in `/tmp`:

```bash
conda run -n link-prediction python -m scripts.wiki.build_graph
conda run -n link-prediction python -m scripts.wiki.fetch_text
conda run -n link-prediction python -m scripts.wiki.build_dataset
conda run -n link-prediction python -m scripts.wiki.run_experiment
```

The Wiki experiment reads shared feature/model YAML settings from `configs/`
(`--config-dir` selects another directory). It checks cached features against
the split, article text, settings, and file checksums. If you have caches from
an older run without a manifest, pass `--rebuild-features` once. `make pipeline-wiki`
already does this.

To run the controlled sparse benchmark from the existing Wiki-CS-8k files
(no SQL dump or text download required):

```bash
make pipeline-wiki-sparse
```

This writes a versioned benchmark under `data/raw/wiki_cs_8k_sparse20/`, caches
training/node features under `data/interim/wiki_cs_8k_sparse20/`, saves models
under `outputs/checkpoints/wiki_cs_8k_sparse20/`, and evaluates all withheld
links in separate random- and hard-negative suites. Direct invocation is:

```bash
conda run -n link-prediction python -m scripts.wiki.build_dataset \
  --protocol sparse-holdout --edge-retention 0.2
conda run -n link-prediction python -m scripts.wiki.run_experiment \
  --benchmark-manifest data/raw/wiki_cs_8k_sparse20/benchmark.json \
  --batch-size 10000
```

To measure the effect of training on difficult negatives, run:

```bash
make pipeline-wiki-sparse-mixed
```

This creates a second balanced training set whose negative class is split
between uniform verified non-links and verified two-hop non-links. It preserves
the observed positive graph and both test files exactly, so the result is a
controlled training-data comparison. The run rebuilds pair-level structural
features and every model. It reuses TF-IDF, POS, and embedding node features
only after checking the article-text hash, feature settings, source-code hashes,
node coverage, and cache-file checksums. Its outputs use the
`wiki_cs_8k_sparse20_mixed` suffix and do not replace the random-training run.

## Running on Kaggle

`notebooks/kaggle_full_pipeline.ipynb` still refers to the former script paths and
needs updating before it can be run. The current command-line entry points above
are the supported path for this branch. When the notebook is updated, it will need:

1. **Accelerator:** GPU (the sentence-transformer encoder uses it).
2. **Internet:** on — required for `git clone`, `pip install`, and the NLTK/HuggingFace downloads.
3. **Input:** attach the `dsaa-2023-competition` competition data.
4. **Secret:** Add-ons → Secrets → `GITHUB_PAT`, a GitHub token with `repo` read scope. The
   notebook clones this repo, so any local change must be pushed before it will be picked up.

The former notebook symlinked `data/raw/dsaa`, `data/interim/dsaa` and
`outputs/checkpoints/dsaa` onto Kaggle paths. Its Python module calls must be
updated before restoring that workflow.
