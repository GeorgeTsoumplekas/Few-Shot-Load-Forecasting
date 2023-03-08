"""Creation of train and test sets of the aggregated time series'.

First, an appropriate size and start month are randomly selected for each aggregated timeseries.
Then, a number of individual time series' are sampled from the pool of the candidate time series
and then aggregated. Each aggregated time series is then saved as a .csv file.

Typical usage:
python3 aggregate.py --config "/path/to/config.yaml"
"""

import argparse
import os
import yaml

import numpy as np
import pandas as pd


def aggregate_timeseries(ts_codes: np.ndarray,
                         source_dir: str,
                         start_month: str,
                         ts_length: int,
                         ) -> pd.DataFrame:
    """Aggregation of the selected time series'.

    The appropriate slice of each time series is selected based on the start month and length of
    the aggregated time series. Then the individual time series' are all added together.

    Args:
        ts_codes: An numpy array that contains the ids of the individual time series' to be
            aggregated.
        source_dir: A string that defines the directory in which the individual time series'
            are stored
        start_month: An integer that defines the first month from which the aggregated time series
            starts.
        ts_length: An integer that defines the number of months contained in the aggregated
            time series
    Returns:
        A pandas DataFrame that contains the aggregated time series.
    """

    aggregated = 0
    for ts_code in ts_codes:
        filename = source_dir + ts_code + '.csv'
        data = pd.read_csv(filename, index_col=0)

        start_date = start_month + '-01 00:00:00'

        # Subtract 1 measurement (=15 minutes) to get the last measurement of the last month
        # contained in the aggregated time series.
        end_date = str(pd.to_datetime(start_date) +
                        pd.DateOffset(months=ts_length) -
                        pd.DateOffset(minutes=15))

        # Data of the individual time series inside the time range of the aggregated time series
        sliced_data = data.loc[start_date:end_date]

        aggregated = aggregated + sliced_data
    return aggregated


def create_set(set_min_length: int,
               set_max_length: int,
               set_size: int,
               set_metadata_dir: str,
               num_timeseries: int,
               source_dir: str,
               dest_dir: str,
               rng: np.random.Generator
               ) -> None:
    """Create a set of aggregated time series'.

    Initially, the start date and length of each aggregated time series is selected randomly.
    Based on these, and by looking through the appropriate segments' metadata, a spcific
    number of valid time series' are randomly selected and aggregated together. Finally, the
    aggregated time series' are saved.

    Args:
        set_min_length: An integer that defines the minimum number of months possibly contained
            in an aggregated time series of the set.
        set_max_length: An integer that defines the maximum number of months possibly contained
            in an aggregated time series of the set.
        set_size: An integer that defines the number of aggregated time series in the set.
        set_metadata_dir: A string defining the path to the directory that contains the segment
            metadata of the set.
        num_timeseries: An integer that defines the number of individual time series to be
            aggregated.
        source_dir: A string that defines the directory in which the individual time series'
            are stored.
        dest_dir: A string that defines the directory in which the aggregated time series' will
            be saved.
        rng: A numpy random generator used for reproducibility purposes.
    """

    # Random length of each aggregated time series in months
    ts_lengths = rng.integers(set_min_length, set_max_length, set_size, endpoint=True)

    # Column index in the segments metadata file that corresponds to the randomly chosen
    # start month of the aggregated time series
    column_indices = np.empty((set_size,), dtype=int)
    for i, _ in enumerate(column_indices):
        column_indices[i] = rng.integers(1, 18-ts_lengths[i], 1, endpoint=True)

    for i, (ts_length, column_idx)in enumerate(zip(ts_lengths, column_indices)):
        # Get the segment metadata which correspond to the length of the aggregated time series
        metadata_filename = set_metadata_dir + str(ts_length).zfill(2) + '_months.csv'
        segments_metadata = pd.read_csv(metadata_filename, index_col=0)

        start_month = (segments_metadata.columns.to_list())[column_idx]

        # Individual time series that have valid data starting from the specified month and that
        # are at least as long as the length of the aggregated time series
        valid_idx = (segments_metadata.loc[segments_metadata[start_month] == 1]).index.to_numpy()

        # Sample from the pool of candidate time series'
        sampled_ts_codes = rng.choice(valid_idx, num_timeseries, replace=False)

        # Aggregate individual time series'
        aggregated = aggregate_timeseries(sampled_ts_codes,
                                          source_dir,
                                          start_month,
                                          ts_length)

        # Save aggregated time series
        agg_filename = dest_dir + str(i).zfill(3) + '.csv'
        aggregated.to_csv(agg_filename, index=False)


def main() -> None:
    """
    Initially, parsing the command line arguments that define the configurations based on
    which the aggregated time series' and train and test sets are created and then creating them.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-C', dest='config_filepath')
    args = parser.parse_args()

    config_filepath = args.config_filepath
    with open(config_filepath, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    source_train_dir = config['source_train_dir']
    source_test_dir = config['source_test_dir']
    dest_train_dir = config['dest_train_dir']
    dest_test_dir = config['dest_test_dir']
    train_metadata_dir = config['train_metadata_dir']
    test_metadata_dir = config['test_metadata_dir']
    train_min_length = config['train_min_length']
    train_max_length = config['train_max_length']
    test_min_length = config['test_min_length']
    test_max_length = config['test_max_length']
    num_timeseries = config['num_timeseries']
    train_size = config['train_size']
    test_size = config['test_size']
    random_seed = config['random_seed']

    # Instantiate random generator for reproducibility purposes
    rng = np.random.default_rng(random_seed)

    # Create train and test set directories
    if not os.path.exists(dest_train_dir):
        os.makedirs(dest_train_dir)

    if not os.path.exists(dest_test_dir):
        os.makedirs(dest_test_dir)

    # Create train set
    create_set(train_min_length,
               train_max_length,
               train_size,
               train_metadata_dir,
               num_timeseries,
               source_train_dir,
               dest_train_dir,
               rng)

    # Create test set
    create_set(test_min_length,
               test_max_length,
               test_size,
               test_metadata_dir,
               num_timeseries,
               source_test_dir,
               dest_test_dir,
               rng)


if __name__ == "__main__":
    main()
