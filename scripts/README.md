# Script entry points

Run from the repository root with `python -m scripts.<group>.<name>`, or use
`make run SCRIPT=<group>.<name>` to use the project Conda environment.
`src/` contains reusable code; `scripts/` contains commands. All outputs below
are relative to the repository root. See [commands](../docs/commands.md) for
Make targets and common overrides.

## DSAA 2023 (`scripts.dsaa`)

| Module | Purpose | Main output |
|---|---|---|
| `download` | Fetch competition data | `data/raw/dsaa/` |
| `compute_structural` | Compute graph heuristics from the original DSAA graph | `data/interim/dsaa/structural_*.csv` |
| `compute_semantic` | Compute TF-IDF, sentence similarity, and POS features | `data/interim/dsaa/{tfidf_*,sentence_emb_*,pos_*}` |
| `label_difficulty` | Label pair difficulty and record thresholds | `data/interim/dsaa/difficulty_*` |
| `train` | Train and validate one selected model (`model=cascade`, etc.) | `outputs/checkpoints/dsaa/`, `outputs/predictions/dsaa/` |
| `predict_test` | Predict unlabelled competition test pairs | `outputs/predictions/dsaa/*_submission.csv` |
| `audit_pairs` | Check duplicate and overlapping pairs | `data/interim/dsaa/leakage_pairs.csv` |
| `audit_negative_sampling` | Test whether negative sampling creates easy shortcuts | `outputs/stats/negative_sampling_audit.json` |
| `audit_prediction_shortcut` | Compare validation predictions with the first-endpoint shortcut | `outputs/stats/hub_in_predictions_audit.json` |
| `audit_protocol` | Quantify legacy graph leakage and endpoint-swap sensitivity | `outputs/stats/supervisor_audit.json` |
| `analyze_hard_residual` | Inspect pairs routed to the last cascade tier | `outputs/predictions/dsaa/hard_residual_analysis.json` |
| `sweep_routing_thresholds` | Measure sensitivity to routing thresholds | `outputs/predictions/dsaa/cascade_threshold_ablation.csv` |
| `benchmark_inference` | Time prediction using cached features | `outputs/predictions/dsaa/throughput_benchmark.json` |
| `compare_matched_samples` | Compare models on identical 20k-pair samples | `outputs/predictions/graph_holdout_v1/matched_samples_v2/` |
| `kaggle` | Submit predictions or retrieve competition scores | `outputs/predictions/dsaa/kaggle_scores.csv` |

The feature and model commands use Hydra `key=value` overrides. The legacy DSAA
training path and the graph-holdout matched comparison have different protocols;
the latter is a controlled comparison within the same artifact-affected dataset.

## Wikipedia (`scripts.wiki`)

| Module | Purpose | Main output |
|---|---|---|
| `build_graph` | Read MediaWiki SQL dumps and select the article graph | `data/raw/wiki_cs_8k/positive_edges.csv` and crawl statistics |
| `fetch_text` | Fetch selected article text | `data/raw/wiki_cs_8k/nodes.tsv` |
| `build_dataset` | Create labelled pairs with sampled negatives | `data/raw/wiki_cs_8k/train.csv` |
| `run_experiment` | Extract features, train, and evaluate on the held-out pairs | `outputs/stats/wiki_cs_8k_experiment_results.json` |

The SQL parser used by `build_graph` is library code in
`src/data/mediawiki_sql.py`, not a separate command.

## Thesis (`scripts.thesis`)

| Module | Purpose | Main output |
|---|---|---|
| `compute_summary_stats` | Aggregate completed experiments | `latex/shared/results/summary_stats.json` |
| `generate_macros` | Render result macros | `latex/shared/generated_macros.tex` |
| `generate_figures` | Render vector figures | `latex/shared/figures/*.pdf` |

Run `make build-thesis` after the DSAA and Wikipedia results and Kaggle score
log are available. The Kaggle notebook still uses old entry points and is left
for its planned rewrite.
