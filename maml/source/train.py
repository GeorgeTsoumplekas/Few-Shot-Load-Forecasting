#!/usr/bin/env python3

import argparse
import os
import yaml

import engine


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
    print(f"Train_losses: {train_losses}")


if __name__ == "__main__":
    main()
