ENV = link-prediction
RUN = conda run -n $(ENV)

.PHONY: env env-update download jupyter pdf clean help kaggle-submit kaggle-check

help:
	@echo "env                  	create conda environment"
	@echo "env-update           	update conda environment from environment.yml"
	@echo "download             	download DSAA 2023 dataset from Kaggle"
	@echo "features-structural  	compute structural features for all pairs"
	@echo "features-semantic    	compute semantic features for all pairs"
	@echo "audit-leakage        	train/test leakage and self-loop audit"
	@echo "analyze-dataset     		separability characterization (trivial-pair fractions)"
	@echo "train MODEL=<name>  		train a model (structural|tfidf|pos|embedding|cascade)"
	@echo "evaluate MODEL=<name> 	generate test set predictions"
	@echo "kaggle-submit FILE=<path> MSG=<text>  submit a predictions CSV to Kaggle"
	@echo "kaggle-check         	poll recent Kaggle submissions and scores"
	@echo "pdf                 		compile the thesis PDF"
	@echo "clean               		remove LaTeX auxiliary files (keeps main.pdf)"
	@echo "jupyter             		start JupyterLab"

env:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

download:
	$(RUN) python -m scripts.download_data

features-structural:
	$(RUN) python -m scripts.compute_structural

features-semantic:
	$(RUN) python -m scripts.compute_semantic

audit-leakage:
	$(RUN) python -m scripts.audit_leakage

analyze-dataset:
	$(RUN) python -m scripts.analyze_dataset

train:
	$(RUN) python -m scripts.train --model $(MODEL)

evaluate:
	$(RUN) python -m scripts.evaluate --model $(MODEL)

kaggle-submit:
	$(RUN) python -m scripts.submit_kaggle --file $(FILE) --message "$(MSG)"

kaggle-check:
	$(RUN) python -m scripts.submit_kaggle --check

pdf:
	cd paper && xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex

clean:
	cd paper && rm -f *.aux *.log *.bbl *.blg *.bcf *.run.xml *.out *.toc *.lof *.lot *.idx *.ilg *.ind \
	  front_matter/*.aux back_matter/*.aux body_matter/*.aux

jupyter:
	$(RUN) jupyter lab
