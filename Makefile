ENV = link-prediction
RUN = conda run -n $(ENV)

.PHONY: env env-update download preprocess jupyter pdf clean help

help:
	@echo "env                  create conda environment"
	@echo "env-update           update conda environment from environment.yml"
	@echo "download             download DSAA 2023 dataset from Kaggle"
	@echo "preprocess           clean and preprocess raw nodes (data/raw → data/interim)"
	@echo "features-structural  compute structural features for all pairs"
	@echo "features-semantic    compute semantic features for all pairs"
	@echo "train MODEL=<name>   train a model (structural|tfidf|pos|embedding|cascade)"
	@echo "evaluate MODEL=<name> generate test set predictions"
	@echo "pdf                  compile the thesis PDF"
	@echo "clean                remove LaTeX auxiliary files (keeps main.pdf)"
	@echo "jupyter              start JupyterLab"

env:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

download:
	$(RUN) python -m scripts.download_data

preprocess:
	$(RUN) python -m scripts.preprocess

features-structural:
	$(RUN) python -m scripts.compute_structural

features-semantic:
	$(RUN) python -m scripts.compute_semantic

train:
	$(RUN) python -m scripts.train --model $(MODEL)

evaluate:
	$(RUN) python -m scripts.evaluate --model $(MODEL)

pdf:
	cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

clean:
	cd paper && rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.idx *.ilg *.ind \
	  front_matter/*.aux back_matter/*.aux body_matter/*.aux

jupyter:
	$(RUN) jupyter lab
