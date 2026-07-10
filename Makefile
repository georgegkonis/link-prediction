ENV = link-prediction
RUN = conda run -n $(ENV)

.PHONY: help \
        env env-update \
        data-download features-structural features-semantic \
        analyze-leakage analyze-dataset analyze-cascade analyze \
        train evaluate \
        kaggle-submit kaggle-check \
        paper-macros paper-figures paper-assets paper-compile paper-clean \
        jupyter

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

analyze: analyze-leakage analyze-dataset analyze-cascade

train:
	$(RUN) python -m scripts.train --model $(MODEL)

evaluate:
	$(RUN) python -m scripts.evaluate --model $(MODEL)

kaggle-submit:
	$(RUN) python -m scripts.submit_kaggle --file $(FILE) --message "$(MSG)"

kaggle-check:
	$(RUN) python -m scripts.submit_kaggle --check

paper-macros:
	$(RUN) python -m scripts.paper.generate_macros

paper-figures:
	$(RUN) python -m scripts.paper.generate_figures

paper-assets: paper-macros paper-figures

paper-compile:
	cd paper && xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex

paper-clean:
	cd paper && rm -f *.aux *.log *.bbl *.blg *.bcf *.run.xml *.out *.toc *.lof *.lot *.idx *.ilg *.ind \
	  front_matter/*.aux back_matter/*.aux body_matter/*.aux

jupyter:
	$(RUN) jupyter lab
