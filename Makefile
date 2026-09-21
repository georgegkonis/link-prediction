ENV = link-prediction
RUN = conda run -n $(ENV) --no-capture-output

.PHONY: help env env-update run train evaluate kaggle-submit kaggle-check latex-compile latex-clean test jupyter

help:
	@echo "Usage:"
	@echo "  make env / env-update                 Manage conda environment"
	@echo "  make run SCRIPT=data.download_data    Run any python script in scripts/"
	@echo "  make train MODEL=cascade              Train a DSAA model"
	@echo "  make evaluate MODEL=cascade           Evaluate a DSAA model"
	@echo "  make kaggle-submit FILE=.. MSG=..     Submit predictions to Kaggle"
	@echo "  make kaggle-check                     Check recent Kaggle scores"
	@echo "  make latex-compile DOC=thesis         Compile thesis/paper/presentation PDF"
	@echo "  make latex-clean DOC=thesis           Clean LaTeX aux files"
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

kaggle-submit:
	$(RUN) python -m scripts.analysis.submit_dsaa_kaggle --file $(FILE) --message "$(MSG)"

kaggle-check:
	$(RUN) python -m scripts.analysis.submit_dsaa_kaggle --check

latex-compile:
	cd latex/$(DOC) && xelatex $(DOC).tex && (biber $(DOC) || true) && xelatex $(DOC).tex && xelatex $(DOC).tex

latex-clean:
	cd latex/$(DOC) && rm -f *.aux *.log *.bbl *.blg *.bcf *.run.xml *.out *.toc *.lof *.lot *.idx *.ilg *.ind \
	  front_matter/*.aux back_matter/*.aux body_matter/*.aux *.nav *.snm *.vrb

test:
	$(RUN) python -m pytest $(PYTEST_ARGS)

jupyter:
	$(RUN) jupyter lab
