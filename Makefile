# Usage:
# env: create virtual environment
# install: install depencencies
# preprocess: preprocess individual raw time series
# aggregate: aggregate idividual preprocessed time series
# train_task_invariant: full pipeline of task invariant LSTM
# inference_task_invariant: inference pipeline of task invariant LSTM
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

aggregate: data_aggregation/pool_config.yaml data_aggregation/aggregate_config.yaml
	./data_aggregation/data_aggregation

train_task_invariant: task_invariant_lstm/config.yaml data/iONA_train_aggregated/* data/iONA_test_aggregated/*
	./task_invariant_lstm/train_script

inference_task_invariant: task_invariant_lstm/inference_config.yaml data/iONA_test_aggregated/*
	./task_invariant_lstm/inference_script

experiment_2: task_specific_lstm/config.yaml data/iONA_test_aggregated/*
	./task_specific_lstm/train_script

MAML: maml/config.yaml data/iONA_train_aggregated/* data/iONA_test_aggregated/*
	./maml/train_script

MAML_inf: maml/inference_config.yaml data/iONA_test_aggregated/*
	./maml/inference_script

embedding: task_embedding/config.yaml data/iONA_train_aggregated/* data/iONA_test_aggregated/*
	./task_embedding/train_script

HSML: hsml/config.yaml data/iONA_train_aggregated/* data/iONA_test_aggregated/* data/iONA_train_embeddings.json data/iONA_test_embeddings.json
	./hsml/train_script

embedding_HSML: task_embedding/config.yaml hsml/config.yaml data/iONA_train_aggregated/* data/iONA_test_aggregated/*
	./task_embedding/train_script
	./hsml/update_config
	./hsml/train_script

.PHONY: clean
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf