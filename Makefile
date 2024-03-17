ENV = link-prediction
RUN = conda run -n $(ENV)

.PHONY: env env-update download preprocess jupyter help

help:
	@echo "env          create conda environment"
	@echo "env-update   update conda environment from environment.yml"
	@echo "download     download DSAA 2023 dataset from Kaggle"
	@echo "preprocess   clean and preprocess raw nodes (data/raw → data/interim)"
	@echo "jupyter      start JupyterLab"

env:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

download:
	$(RUN) python -m scripts.download_data

preprocess:
	$(RUN) python -m scripts.preprocess

jupyter:
	$(RUN) jupyter lab
