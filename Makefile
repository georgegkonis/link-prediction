ENV = link-prediction
RUN = conda run -n $(ENV) --no-capture-output

.PHONY: help env env-update run train predict-test dsaa-train-all compare-matched kaggle-submit kaggle-check compute-stats thesis-macros thesis-figures thesis-assets build-thesis pipeline-dsaa pipeline-wiki latex-compile latex-clean test jupyter

help:
	@echo "Usage:"
	@echo "  make env / env-update                 Manage conda environment"
	@echo "  make run SCRIPT=dsaa.download         Run any Python entry point in scripts/"
	@echo "  make train MODEL=cascade              Train a DSAA model"
	@echo "  make predict-test MODEL=cascade       Write DSAA test predictions"
	@echo "  make dsaa-train-all                   Train all six DSAA models"
	@echo "  make compare-matched                  Compare DSAA models on equal 20k samples"
	@echo "  make kaggle-submit FILE=.. MSG=..     Submit predictions to Kaggle (WAIT=--wait optional)"
	@echo "  make kaggle-check                     Check recent Kaggle scores (WAIT=--wait optional)"
	@echo "  make thesis-assets                    Regenerate committed macros and vector figures"
	@echo "  make build-thesis                     Build thesis from existing experiment results"
	@echo "  make latex-compile DOC=thesis         Compile thesis/paper/presentation PDF"
	@echo "  make latex-clean DOC=thesis           Clean LaTeX aux files"
	@echo "  make pipeline-dsaa                    Rebuild local DSAA results (Kaggle scores are external)"
	@echo "  make pipeline-wiki                    Rebuild Wiki results (requires SQL dumps and text access)"
	@echo "  make test                             Run tests"

env:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

run:
	$(RUN) python -m scripts.$(SCRIPT)

train:
	$(RUN) python -m scripts.dsaa.train model=$(MODEL)

predict-test:
	$(RUN) python -m scripts.dsaa.predict_test model=$(MODEL)

dsaa-train-all:
	$(MAKE) train MODEL=structural
	$(MAKE) train MODEL=tfidf
	$(MAKE) train MODEL=pos
	$(MAKE) train MODEL=embedding
	$(MAKE) train MODEL=svm
	$(MAKE) train MODEL=cascade

compare-matched:
	$(RUN) python -m scripts.dsaa.compare_matched_samples

kaggle-submit:
	$(RUN) python -m scripts.dsaa.kaggle --file $(FILE) --message "$(MSG)" $(WAIT)

kaggle-check:
	$(RUN) python -m scripts.dsaa.kaggle --check $(WAIT)

compute-stats:
	$(RUN) python -m scripts.thesis.compute_summary_stats

thesis-macros:
	$(RUN) python -m scripts.thesis.generate_macros

thesis-figures:
	MPLCONFIGDIR=/tmp/matplotlib-link-prediction $(RUN) python -m scripts.thesis.generate_figures

thesis-assets: thesis-macros thesis-figures

latex-compile:
	cd latex/$(DOC) && xelatex -interaction=nonstopmode -halt-on-error $(DOC).tex && (biber $(DOC) || true) && xelatex -interaction=nonstopmode -halt-on-error $(DOC).tex && xelatex -interaction=nonstopmode -halt-on-error $(DOC).tex

latex-clean:
	cd latex/$(DOC) && rm -f *.aux *.log *.bbl *.blg *.bcf *.run.xml *.out *.toc *.lof *.lot *.idx *.ilg *.ind \
	  front_matter/*.aux back_matter/*.aux body_matter/*.aux *.nav *.snm *.vrb

test:
	$(RUN) python -m pytest $(PYTEST_ARGS)

jupyter:
	$(RUN) jupyter lab

# Full Automation Pipelines
pipeline-dsaa:
	@echo "--- Rebuilding local DSAA results; Kaggle scores require a separate submission ---"
	$(RUN) python -m scripts.dsaa.download
	$(RUN) python -m scripts.dsaa.compute_structural
	$(RUN) python -m scripts.dsaa.compute_semantic
	$(RUN) python -m scripts.dsaa.audit_pairs
	$(RUN) python -m scripts.dsaa.label_difficulty
	$(RUN) python -m scripts.dsaa.audit_negative_sampling
	$(MAKE) dsaa-train-all
	$(MAKE) predict-test MODEL=cascade
	$(RUN) python -m scripts.dsaa.audit_prediction_shortcut
	$(RUN) python -m scripts.dsaa.audit_protocol --swap
	$(RUN) python -m scripts.dsaa.analyze_hard_residual
	$(RUN) python -m scripts.dsaa.benchmark_inference
	$(RUN) python -m scripts.dsaa.sweep_routing_thresholds
	$(MAKE) compare-matched

pipeline-wiki:
	@echo "--- Running Full Wiki-CS-8k Pipeline ---"
	$(RUN) python -m scripts.wiki.build_graph
	$(RUN) python -m scripts.wiki.fetch_text
	$(RUN) python -m scripts.wiki.build_dataset
	$(RUN) python -m scripts.wiki.run_experiment

build-thesis:
	@echo "--- Compiling Thesis Assets & PDF ---"
	$(MAKE) compute-stats
	$(MAKE) thesis-assets
	$(MAKE) latex-compile DOC=thesis
