# Usage:
# env: create virtual environment
# install: install depencencies
# preprocess: preprocess individual raw time series
# aggregate: aggregate idividual preprocessed time series
# experiment_1: execute task invariant lstm
# experiment_2: execute task specific lstm
# MAML: execute MAML model

VENV_PATH='.venv/bin/activate'

env:
	python3 -m venv .venv
	source $(VENV_PATH)

install: requirements.txt
	pip install -r requirements.txt

preprocess: data_preprocessing/preprocess_config.yaml data_preprocessing/slice_config.yaml
	./data_preprocessing/data_preprocessing

aggregate: data_aggregation/aggregate_config.yaml
	./data_aggregation/data_aggregation

experiment_1: task_invariant_lstm/config.yaml data/iONA_train_aggregated/* data/iONA_test_aggregated/*
	./task_invariant_lstm/train_script

experiment_2: task_specific_lstm/config.yaml data/iONA_train_aggregated/* data/iONA_test_aggregated/*
	./task_specific_lstm/train_script

MAML: maml/config.yaml data/mini_iONA_train_aggregated/* data/mini_iONA_test_aggregated/*
	./maml/train_script

.PHONY: clean
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf