#!/usr/bin/env python3

import argparse
import os
import warnings
import yaml

import optuna
from sklearn.model_selection import KFold
import torch

import engine

# Suppress warning that occurs due to accessing a non-leaf tensor
warnings.filterwarnings("ignore")


def objective(trial, ht_config, data_config, data_filenames):
    # Hyperparameter search space
    config = {
        'kfolds': ht_config['k_folds'],
        'seed': ht_config['seed'],
        'task_batch_size': ht_config['task_batch_size'],
        'sample_batch_size': ht_config['sample_batch_size'],
        'num_inner_steps': ht_config['num_inner_steps'],
        'eta_min': ht_config['eta_min'],
        'train_epochs': trial.suggest_int(
            'train_epochs',
            int(ht_config['train_epochs']['lower_bound']),
            int(ht_config['train_epochs']['uppper_bound'])
        ),
        'lstm_hidden_units': trial.suggest_int(
            'lstm_hidden_units',
            int(ht_config['lstm_hidden_units']['upper_bound']),
            int(ht_config['lstm_hidden_units']['upper_bound']),
            log=ht_config['lstm_hidden_units']['log']
        ),
        'init_learning_rate': trial.suggest_float(
            'init_learning_rate',
            float(ht_config['init_learning_rate']['lower_bound']),
            float(ht_config['init_learning_rate']['upper_bound']),
            log=ht_config['init_learning_rate']['log']
        ),
        'meta_learning_rate': trial.suggest_float(
            'meta_learning_rate',
            float(ht_config['meta_learning_rate']['lower_bound']),
            float(ht_config['meta_learning_rate']['upper_bound']),
            log=ht_config['meta_learning_rate']['log']
        )
    }

    # Set random seed for reproducibility purposes
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    results = {}  # Contains the validation loss of each fold

    k_folds = config['kfolds']

    # Cross-validation splits - they should be random since we are splitting the tasks
    # (non-sequential) and not a single timeseries (sequential)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_idx,val_idx) in enumerate(kfold.split(data_filenames)):
        meta_learner = engine.build_meta_learner(opt_config=config['opt_config'],
                                                data_config=config['data_config'])


def hyperparameter_tuning():
    pass


# TODO: change appropriately
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', dest='train_dir')
    parser.add_argument('--test_dir', dest='test_dir')
    parser.add_argument('--config', '-C', dest='config_filepath')
    args = parser.parse_args()

    train_dir = args.train_dir
    test_dir = args.test_dir
    config_filepath = args.config_filepath
    with open(config_filepath, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    # Paths to time series used for training
    train_filenames = []
    for root, _, files in os.walk(train_dir, topdown=False):
        for file in files:
            filename = root + file
            train_filenames.append(filename)

    # Paths to time series used for testing
    test_filenames = []
    for root, _, files in os.walk(test_dir, topdown=False):
        for file in files:
            filename = root + file
            test_filenames.append(filename)

    meta_learner = engine.build_meta_learner(opt_config=config['opt_config'],
                                             data_config=config['data_config'])
    
    train_losses = meta_learner.meta_train(train_filenames)
    print(f"\nTrain_losses: {train_losses}")


if __name__ == "__main__":
    main()
