# Commands

All `make` targets invoke `conda run -n link-prediction` internally — no manual activation needed.

## Make Targets

```bash
make env                               # Create conda environment
make env-update                        # Update from environment.yml

# Run any script via 'make run SCRIPT=...'
make run SCRIPT=data.download_data     # Pull DSAA 2023 dataset from Kaggle
make run SCRIPT=data.compute_structural # Compute structural features
make run SCRIPT=data.compute_semantic  # Compute semantic features
make run SCRIPT=analysis.analyze_dataset # Run separability characterization
make run SCRIPT=paper.generate_figures # Generate thesis figures

# Dedicated Training Targets
make train MODEL=structural            # Train a baseline model
make train MODEL=cascade               # Train CascadeLP
make evaluate MODEL=cascade            # Run inference on test set
make compare-matched                   # Equal-size DSAA comparison (3 x 20k samples)

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
conda run -n link-prediction python -m scripts.data.compute_semantic --nrows 300 --skip-st
```

## Wiki-CS-8k Pipeline

For the Wiki-CS-8k dataset, the following parallel pipeline of scripts is used:

```bash
conda run -n link-prediction python -m scripts.data.build_from_wikidump
conda run -n link-prediction python -m scripts.data.fetch_wiki_cs_8k_text
conda run -n link-prediction python -m scripts.data.build_wiki_cs_8k_dataset
conda run -n link-prediction python -m scripts.analysis.run_wiki_cs_8k_experiment
```

## Running on Kaggle

`notebooks/kaggle_full_pipeline.ipynb` runs the whole pipeline (features → analysis → train all
six models → evaluate → submission) inside a Kaggle notebook session. Import it, then:

1. **Accelerator:** GPU (the sentence-transformer encoder uses it).
2. **Internet:** on — required for `git clone`, `pip install`, and the NLTK/HuggingFace downloads.
3. **Input:** attach the `dsaa-2023-competition` competition data.
4. **Secret:** Add-ons → Secrets → `GITHUB_PAT`, a GitHub token with `repo` read scope. The
   notebook clones this repo, so any local change must be pushed before it will be picked up.

The notebook is a thin driver — it symlinks `data/raw/dsaa`, `data/interim/dsaa` and `outputs/checkpoints/dsaa`
onto Kaggle paths and then calls the same `python -m scripts.…` entry points as the make targets.
Set `SAMPLE_ROWS` in the first cell to truncate `train.csv`/`test.csv` for a minutes-long smoke
test before committing to a multi-hour full run.
