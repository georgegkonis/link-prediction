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

The download, feature, and model commands use Hydra `key=value` overrides.
`paths.raw` and `paths.interim` control feature input/output locations; a
development run writes under `<paths.interim>/dev_.../`. Feature commands
and model runs save their resolved settings beside their outputs. The legacy DSAA
training path and the graph-holdout matched comparison have different protocols;
the latter is a controlled comparison within the same artifact-affected dataset.
The protocol audit, threshold sweep, and inference benchmark reuse the recorded
CascadeLP split and paths when a run snapshot is present. Their default paths
remain available for older results that predate snapshots.

## Wikipedia (`scripts.wiki`)

| Module | Purpose | Main output |
|---|---|---|
| `build_graph` | Read MediaWiki SQL dumps and select the article graph | `data/raw/wiki_cs_8k/positive_edges.csv` and crawl statistics |
| `fetch_text` | Fetch selected article text | `data/raw/wiki_cs_8k/nodes.tsv` |
| `build_dataset` | Create the balanced dataset or a sparse graph holdout with random/hard negatives | `data/raw/wiki_cs_8k/train.csv` or `data/raw/wiki_cs_8k_sparse20/benchmark.json` |
| `run_experiment` | Train and evaluate the original or manifest-defined sparse benchmark | `outputs/stats/wiki_cs_8k_experiment_results.json` or `wiki_cs_8k_sparse20_results.json` |

The SQL parser used by `build_graph` is library code in
`src/data/mediawiki_sql.py`, not a separate command.
Wiki commands use `argparse`; `run_experiment` reads feature/model defaults from
the shared `configs/` YAML directory. It verifies a feature-cache manifest
before reuse. For caches made before manifests were introduced, use
`--rebuild-features` once or select a fresh `--directory`.

`make pipeline-wiki-sparse` reuses the complete verified Wiki-CS-8k graph. It
retains 20% of links as the observed training graph, uses every other link for
testing, and reports separate random-nonedge and two-hop-hard-nonedge results.
`make pipeline-wiki-sparse-mixed` keeps those test suites fixed while replacing
half of the uniform training non-links with verified two-hop non-links. It
stores the second dataset, feature cache, checkpoints, predictions, and results
under paths ending in `wiki_cs_8k_sparse20_mixed`. Test features are assembled
in bounded batches; CascadeLP requests POS and embedding pair features only for
pairs that reach the corresponding tier.

## Thesis (`scripts.thesis`)

| Module | Purpose | Main output |
|---|---|---|
| `compute_summary_stats` | Aggregate completed experiments | `latex/shared/results/summary_stats.json` |
| `generate_macros` | Render result macros | `latex/shared/generated_macros.tex` |
| `generate_figures` | Render vector figures | `latex/shared/figures/*.pdf` |

Run `make build-thesis` after the DSAA and Wikipedia results and Kaggle score
log are available.
