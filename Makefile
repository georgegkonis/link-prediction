ENV = link-prediction
RUN = conda run -n $(ENV) --no-capture-output

.PHONY: help \
        env env-update \
        data-download features-structural features-semantic \
        analyze-leakage analyze-dataset analyze-cascade analyze-node2vec \
        analyze-hard-residual benchmark-throughput ablate-thresholds analyze \
        train evaluate \
        kaggle-submit kaggle-check \
        paper-macros paper-figures paper-assets paper-compile paper-clean \
        test jupyter

# End-to-end order to fully populate data/interim/ + outputs/ before paper-assets
# (figure/macro generation itself never re-runs any of this — see generate_macros.py /
# generate_figures.py docstrings):
#   data-download
#   → features-structural, features-semantic
#   → analyze-leakage, analyze-dataset
#   → train MODEL=<structural|tfidf|pos|embedding|svm|cascade>   (cascade defaults to the
#     reproducible heuristics-only checkpoint the thesis reports; pass training.no_n2v=false
#     +tag=n2v for the n2v-ablation variant)
#   → evaluate MODEL=<same 6>
#   → ablate-thresholds, analyze-node2vec, analyze-hard-residual, benchmark-throughput
#   → paper-assets

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
	@echo "Paper"
	@echo "  paper-macros                      regenerate generated_macros.tex from val metrics"
	@echo "  paper-figures                     regenerate all thesis figures"
	@echo "  paper-assets                      run paper-macros and paper-figures"
	@echo "  paper-compile                     compile the thesis PDF"
	@echo "  paper-clean                       remove LaTeX auxiliary files (keeps main.pdf)"
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
	$(RUN) python -m scripts.train model=$(MODEL)

evaluate:
	$(RUN) python -m scripts.evaluate model=$(MODEL)

kaggle-submit:
	$(RUN) python -m scripts.submit_kaggle --file $(FILE) --message "$(MSG)"

kaggle-check:
	$(RUN) python -m scripts.submit_kaggle --check

paper-macros:
	$(RUN) python -m scripts.paper.generate_macros

paper-figures:
	$(RUN) python -m scripts.paper.generate_figures

paper-assets: paper-macros paper-figures

paper-version:
	echo '\newcommand{\draftversion}{DRAFT}' > paper/version.tex

paper-compile: paper-version
	cd paper && xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex

paper-clean:
	cd paper && rm -f *.aux *.log *.bbl *.blg *.bcf *.run.xml *.out *.toc *.lof *.lot *.idx *.ilg *.ind \
	  front_matter/*.aux back_matter/*.aux body_matter/*.aux

test:
	$(RUN) python -m pytest $(PYTEST_ARGS)

jupyter:
	$(RUN) jupyter lab
