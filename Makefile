ENV = link-prediction
RUN = conda run -n $(ENV) --no-capture-output

.PHONY: help env env-update run train evaluate compare-matched kaggle-submit kaggle-check compute-stats thesis-macros thesis-figures thesis-assets latex-compile latex-clean test jupyter

help:
	@echo "Usage:"
	@echo "  make env / env-update                 Manage conda environment"
	@echo "  make run SCRIPT=data.download_data    Run any python script in scripts/"
	@echo "  make train MODEL=cascade              Train a DSAA model"
	@echo "  make evaluate MODEL=cascade           Evaluate a DSAA model"
	@echo "  make compare-matched                  Compare DSAA models on equal 20k samples"
	@echo "  make kaggle-submit FILE=.. MSG=..     Submit predictions to Kaggle"
	@echo "  make kaggle-check                     Check recent Kaggle scores"
	@echo "  make thesis-assets                    Regenerate committed macros and vector figures"
	@echo "  make latex-compile DOC=thesis         Compile thesis/paper/presentation PDF"
	@echo "  make latex-clean DOC=thesis           Clean LaTeX aux files"
	@echo "  make pipeline-dsaa                    Run entire DSAA pipeline"
	@echo "  make pipeline-wiki                    Run entire Wiki pipeline"
	@echo "  make pipeline-paper                   Build stats, figures, and thesis PDF"
	@echo "  make pipeline-all                     Run EVERYTHING end-to-end"
	@echo "  make test                             Run tests"

env:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

run:
	$(RUN) python -m scripts.$(SCRIPT)

train:
	$(RUN) python -m scripts.analysis.run_dsaa_train model=$(MODEL)

evaluate:
	$(RUN) python -m scripts.analysis.run_dsaa_evaluate model=$(MODEL)

compare-matched:
	$(RUN) python -m scripts.analysis.compare_matched_samples

kaggle-submit:
	$(RUN) python -m scripts.analysis.submit_dsaa_kaggle --file $(FILE) --message "$(MSG)"

kaggle-check:
	$(RUN) python -m scripts.analysis.submit_dsaa_kaggle --check

compute-stats:
	$(RUN) python -m scripts.paper.compute_summary_stats

thesis-macros:
	$(RUN) python -m scripts.paper.generate_macros

thesis-figures:
	MPLCONFIGDIR=/tmp/matplotlib-link-prediction $(RUN) python -m scripts.paper.generate_figures

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
	@echo "--- Running Full DSAA Pipeline ---"
	$(RUN) python -m scripts.data.download_data
	$(RUN) python -m scripts.data.compute_structural
	$(RUN) python -m scripts.data.compute_semantic
	$(RUN) python -m scripts.analysis.audit_leakage
	$(RUN) python -m scripts.analysis.analyze_dataset
	$(RUN) python -m scripts.analysis.run_dsaa_train model=cascade
	$(RUN) python -m scripts.analysis.run_dsaa_evaluate model=cascade
	$(RUN) python -m scripts.analysis.analyze_cascade
	$(RUN) python -m scripts.analysis.analyze_hard_residual
	$(RUN) python -m scripts.analysis.benchmark_throughput
	$(RUN) python -m scripts.analysis.ablate_cascade_thresholds

pipeline-wiki:
	@echo "--- Running Full Wiki-CS-8k Pipeline ---"
	$(RUN) python -m scripts.data.build_from_wikidump
	$(RUN) python -m scripts.data.fetch_wiki_cs_8k_text
	$(RUN) python -m scripts.data.build_wiki_cs_8k_dataset
	$(RUN) python -m scripts.analysis.run_wiki_cs_8k_experiment

pipeline-paper:
	@echo "--- Compiling Thesis Assets & PDF ---"
	$(MAKE) compute-stats
	$(MAKE) thesis-assets
	$(MAKE) latex-compile DOC=thesis

pipeline-all: pipeline-dsaa pipeline-wiki pipeline-paper
