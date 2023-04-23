import argparse
import os
import yaml

from utils import set_random_seeds


def objective(trial, ht_config, data_config, data_filenames):
    pass


def hyperparameter_tuning():
    pass


def train_optimal():
    pass


def evaluate_optimal():
    pass


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

    # Create results directory
    results_dir_name = './task_embedding/results/'
    if not os.path.exists(results_dir_name):
        os.makedirs(results_dir_name)

    # Set random seeds for reproducibility purposes
    set_random_seeds(config['seed'])

    # Hyperparameter tuning
    n_trials = config['n_trials']
    ht_config = config['ht_config']
    data_config = config['data_config']

    opt_config = hyperparameter_tuning(n_trials,
                                       results_dir_name,
                                       ht_config,
                                       data_config,
                                       train_filenames)

    # Train optimal task-embedding model
    train_optimal(opt_config,
                       data_config,
                       train_filenames,
                       results_dir_name)

    # Load optimal task-embedding model and evaluate it on test tasks
    evaluate_optimal(opt_config,
                          data_config,
                          test_filenames,
                          results_dir_name)


if __name__ == "__main__":
    main()
