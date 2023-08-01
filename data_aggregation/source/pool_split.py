"""

"""

import argparse
import os
import random
import shutil
import yaml


def split(source_dir, train_test_split):
    """
    
    """

    # Create output directory for pool of train time series'
    train_pool_dir_name = source_dir[:-21] + '_train/'
    if not os.path.exists(train_pool_dir_name):
        os.makedirs(train_pool_dir_name)

    # Create output directory for pool of test time series'
    test_pool_dir_name = source_dir[:-21] + '_test/'
    if not os.path.exists(test_pool_dir_name):
        os.makedirs(test_pool_dir_name)

    print(train_pool_dir_name)
    print(test_pool_dir_name)

    # Seperate train and test time series'
    for _ , _, files in os.walk(source_dir, topdown=False):
        print(len(files))

        train_pool_length = int(train_test_split*len(files))
        train_pool_ts = random.sample(files, train_pool_length)

        # The rest of the time series' belong to the test pool
        test_pool_ts = list(set(train_pool_ts)-set(files))

    print(len(train_pool_ts))
    print(len(test_pool_ts))

    # # Move train time series to new folder
    # for train_ts in train_pool_ts:
    #     shutil.move()

    # # Move test time series to new folder
    # for test_ts in test_pool_ts:
    #     shutil.move()


def main() -> None:
    """
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-C', dest='config_filepath')
    args = parser.parse_args()

    config_filepath = args.config_filepath
    with open(config_filepath, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    source_dir = config['source_dir']
    seed = int(config['seed'])
    train_test_split = float(config['train_test_split'])

    print(source_dir)

    # Set random seed
    random.seed(seed)

    split(source_dir, train_test_split)


if __name__ == "__main__":
    main()
