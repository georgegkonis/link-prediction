ENV = link-prediction
RUN = conda run -n $(ENV) --no-capture-output

.PHONY: help \
        env env-update \
        data-download features-structural features-semantic \
        analyze-leakage analyze-dataset analyze-cascade analyze-node2vec \
        analyze-hard-residual benchmark-throughput ablate-thresholds analyze \
        train evaluate \
        kaggle-submit kaggle-check \
        compute-stats thesis-macros thesis-figures thesis-assets thesis-compile thesis-clean \
        presentation-compile presentation-clean \
        paper-compile paper-clean \
        test jupyter

# End-to-end order to fully populate data/interim/ + outputs/ before thesis-assets
# (thesis-macros/thesis-figures never touch data/raw, data/interim, or outputs/predictions —
# they only read the committed outputs/stats/summary_stats.json. compute-stats is the one step
# that does, and its output is what gets committed — see compute_summary_stats.py's docstring):
#   data-download
#   → features-structural, features-semantic
#   → analyze-leakage, analyze-dataset
#   → train MODEL=<structural|tfidf|pos|embedding|svm|cascade>   (cascade defaults to the
#     reproducible heuristics-only checkpoint the thesis reports; pass training.no_n2v=false
#     +tag=n2v for the n2v-ablation variant)
#   → evaluate MODEL=<same 6>
#   → ablate-thresholds, analyze-node2vec, analyze-hard-residual, benchmark-throughput
#   → compute-stats
#   → thesis-assets

help:
	@echo "Environment"
	@echo "  env                               create conda environment"
	@echo "  env-update                        update from environment.yml"
	@echo ""
	@echo "Data pipeline"
	@echo "  data-download                     download DSAA 2023 dataset from Kaggle"
	@echo "  features-structural               compute CN/Jaccard/AA/PA/Node2Vec features"
	@echo "  features-semantic                 compute TF-IDF, Sentence-Transformer, POS features"
	@echo ""
	@echo "Analysis"
	@echo "  analyze-leakage                   train/test leakage and self-loop audit"
	@echo "  analyze-dataset                   separability characterization (trivial-pair fractions)"
	@echo "  analyze-cascade                   CascadeLP tier and difficulty breakdown"
	@echo "  analyze-node2vec                  Node2Vec with/without ablation + test coverage"
	@echo "  analyze-hard-residual             Tier-3 hard-residual / nodes.tsv join"
	@echo "  benchmark-throughput              CPU inference throughput benchmark"
	@echo "  ablate-thresholds                 tau1/tau2 threshold grid sweep"
	@echo "  analyze                           run all analysis targets"
	@echo ""
	@echo "Model"
	@echo "  train MODEL=<name>                train a model (structural|tfidf|pos|embedding|svm|cascade)"
	@echo "  evaluate MODEL=<name>             generate test set predictions"
	@echo ""
	@echo "Kaggle"
	@echo "  kaggle-submit FILE=<path> MSG=<text>  submit a predictions CSV"
	@echo "  kaggle-check                      poll recent submission scores"
	@echo ""
	@echo "Thesis (latex/thesis/)"
	@echo "  compute-stats                     aggregate data/interim + outputs/predictions into outputs/stats/summary_stats.json"
	@echo "  thesis-macros                     regenerate latex/shared/generated_macros.tex from summary_stats.json"
	@echo "  thesis-figures                    regenerate all figures into outputs/figures/ from summary_stats.json"
	@echo "  thesis-assets                     run thesis-macros and thesis-figures"
	@echo "  thesis-compile                    compile the thesis PDF"
	@echo "  thesis-clean                      remove LaTeX auxiliary files (keeps thesis.pdf)"
	@echo ""
	@echo "Presentation (latex/presentation/)"
	@echo "  presentation-compile              compile the presentation slides PDF"
	@echo "  presentation-clean                remove LaTeX auxiliary files (keeps the PDF)"
	@echo ""
	@echo "Short paper (latex/paper/)"
	@echo "  paper-compile                     compile the short paper PDF"
	@echo "  paper-clean                       remove LaTeX auxiliary files (keeps the PDF)"
	@echo ""
	@echo "Other"
	@echo "  test                              run the unit test suite (no data/ needed)"
	@echo "  jupyter                           start JupyterLab"

env:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

data-download:
	$(RUN) python -m scripts.data.download_data

features-structural:
	$(RUN) python -m scripts.data.compute_structural

features-semantic:
	$(RUN) python -m scripts.data.compute_semantic

analyze-leakage:
	$(RUN) python -m scripts.analysis.audit_leakage

analyze-dataset:
	$(RUN) python -m scripts.analysis.analyze_dataset

analyze-cascade:
	$(RUN) python -m scripts.analysis.analyze_cascade

analyze-node2vec:
	$(RUN) python -m scripts.analysis.ablate_node2vec

analyze-hard-residual:
	$(RUN) python -m scripts.analysis.analyze_hard_residual

benchmark-throughput:
	$(RUN) python -m scripts.analysis.benchmark_throughput

ablate-thresholds:
	$(RUN) python -m scripts.analysis.ablate_cascade_thresholds

analyze: analyze-leakage analyze-dataset analyze-cascade analyze-node2vec analyze-hard-residual benchmark-throughput

train:
	$(RUN) python -m scripts.analysis.run_dsaa_train model=$(MODEL)

evaluate:
	$(RUN) python -m scripts.analysis.run_dsaa_evaluate model=$(MODEL)

kaggle-submit:
	$(RUN) python -m scripts.analysis.submit_dsaa_kaggle --file $(FILE) --message "$(MSG)"

kaggle-check:
	$(RUN) python -m scripts.analysis.submit_dsaa_kaggle --check

compute-stats:
	$(RUN) python -m scripts.paper.compute_summary_stats

thesis-macros:
	$(RUN) python -m scripts.paper.generate_macros

thesis-figures:
	$(RUN) python -m scripts.paper.generate_figures

thesis-assets: thesis-macros thesis-figures

thesis-compile:
	cd latex/thesis && xelatex thesis.tex && biber thesis && xelatex thesis.tex && xelatex thesis.tex

thesis-clean:
	cd latex/thesis && rm -f *.aux *.log *.bbl *.blg *.bcf *.run.xml *.out *.toc *.lof *.lot *.idx *.ilg *.ind \
	  front_matter/*.aux back_matter/*.aux body_matter/*.aux

presentation-compile:
	cd latex/presentation && xelatex presentation.tex

presentation-clean:
	cd latex/presentation && rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb

paper-compile:
	cd latex/paper && xelatex paper.tex && biber paper && xelatex paper.tex && xelatex paper.tex

paper-clean:
	cd latex/paper && rm -f *.aux *.log *.bbl *.blg *.bcf *.run.xml *.out *.toc *.lof *.lot *.idx *.ilg *.ind

test:
	$(RUN) python -m pytest $(PYTEST_ARGS)

jupyter:
	$(RUN) jupyter lab
