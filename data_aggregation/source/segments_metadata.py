"""Segmenting of the preprocessed time series'.

The preprocessed time series' are examined to find out single- and multi-monthly continuous
segments of them. These segmented parts are not stored separately. However, csv files are created
in which we mark down which segments can be found in each time series. These csv files can then be
used as references to sample time series and aggregate them based on what we want.

Typical usage:
python3 segments_metadata.py --data_dir "/path/to/preprocessed_data/"
"""

import argparse
import os

import pandas as pd

# Suppress unnecessary pandas warnings
pd.options.mode.chained_assignment = None


def timeseries_lengths(data_dir):
    """Determine the start and end date of each time series.

    In order to find the beginning and end date of a time series, it is not necessary to read the
    whole csv file, but only its first and last row.

    Args:
        data_dir: Path to the directory that contains the preprocessed time series'.
    Returns:
        A pandas DataFrame that contains the start and end dates of each time series.
    """

    ts_lengths = pd.DataFrame(columns=['file', 'start', 'end'])

    for root, _, files in os.walk(data_dir, topdown=False):
        for file in files:
            filename = root+file

            # Read first row only
            start = pd.read_csv(filename, nrows=1)

            # Read last row only
            with open(filename, "r", encoding='utf8') as stream:
                end = stream.readlines()[-1].strip().split(",")

            ts_lengths.loc[len(ts_lengths)] = [file[:-4], start['DateTime'][0], end[0]]

    # String to DateTime conversion
    ts_lengths['start'] = pd.to_datetime(ts_lengths['start'])
    ts_lengths['end'] = pd.to_datetime(ts_lengths['end'])

    return ts_lengths


def monthly_segments(ts_lengths):
    """Find monthly segments with appropriate data in each time series.

    This is done by examining whether the start and end dates of the examined month are inside the
    beginning and end dates of the time series.

    Args:
        ts_lengths: A pandas DataFrame that contains the start and end dates of each time series.
    Returns:
        A pandas DataFrame with the whole months contained in each time series.
    """

    # Months and years examined
    months = [[str(i).zfill(2) for i in range(10, 13)],
              [str(i).zfill(2) for i in range(1, 13)],
              [str(i).zfill(2) for i in range(1, 5)]]
    years = ['2020', '2021', '2022']

    # The column names correspond to the year-month each segment starts
    column_names = [year + '-' + month
                    for i, year in enumerate(years)
                    for month in months[i]]
    column_names = column_names[:-1]  # Exlcude last month, used only as a max date
    column_names = ['ts_code'] + column_names

    month_starts = [pd.to_datetime(year + '-' + month + '-' + '01 00:00:00')
                for i, year in enumerate(years)
                for month in months[i]]

    one_month_segments = pd.DataFrame(columns=column_names)
    one_month_segments['ts_code'] = ts_lengths['file']

    for index, timeseries in ts_lengths.iterrows():
        start = timeseries['start']
        end = timeseries['end']

        # Examine each possible month
        for i in range(0, len(month_starts)-1):
            month_start = month_starts[i]
            month_end = month_starts[i+1]

            # Check if it is insede the boundaries of the time series
            if (month_start >= start) & (month_end <= end):
                date = month_start.strftime('%Y-%m')
                one_month_segments[date][index] = 1

    one_month_segments.set_index('ts_code', inplace=True)
    one_month_segments.dropna(axis=0, how='all', inplace=True)
    one_month_segments.fillna(0, inplace=True)

    one_month_segments['sum'] = one_month_segments[list(one_month_segments.columns)].sum(axis=1)

    return one_month_segments


def multi_monthly_segments(num_months, one_month_segments):
    """Find multi-monthly segments with appropriate data in each time series.

    This is done by finding consecutive months within the bounds of the time series and marking
    the beginning of each such segment.

    Args:
        num_months: An integer that defines the number of months each segment contains.
        one_month_segments: A pandas DataFrame with the whole months contained in each time series.
    Returns:
        A pandas DataFrame with the multi-month segments contained in each time series.
    """

    # The column names correspond to the year-month each segment starts
    multi_month_segments = pd.DataFrame(columns=one_month_segments.reset_index()
                                        .columns
                                        .values[:-num_months])

    multi_month_segments['ts_code'] = (one_month_segments.reset_index())['ts_code']
    for column_name in multi_month_segments.columns.values[1:]:
        multi_month_segments[column_name] = None
    multi_month_segments.set_index('ts_code', inplace=True)

    # Examine each set of consecutive months
    for i in range(multi_month_segments.shape[1]):
        period = one_month_segments.columns.values[i:i+num_months]

        # Slice of the single month segments that contains the months that belong to the
        # multi-month segment
        slice_df = one_month_segments[period]

        # Check if all the examined months are valid
        slice_df['sum'] = slice_df[list(slice_df.columns)].sum(axis=1)
        ts_codes = slice_df.index[slice_df['sum'] == num_months].tolist()

        # Mark down the time series that contain this multi-month segment
        multi_month_segments.loc[ts_codes, period[0]] = int(1)

    multi_month_segments.dropna(axis=0, how='all', inplace=True)
    multi_month_segments.fillna(int(0), inplace=True)
    multi_month_segments['sum'] = \
        multi_month_segments[list(multi_month_segments.columns)].sum(axis=1)

    return multi_month_segments


def main() -> None:
    """
    Initially, parsing the command line arguments to determine the directory that contains the
    preprocessed time series' and the creating the single- and multi-month segment tables and
    storing them in a separate directory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-D', dest='data_dir')
    args = parser.parse_args()

    data_dir = args.data_dir
    dir_type = (data_dir.split('_'))[1][:-1]

    # Create output directory
    metadata_dir_name = './data/' + dir_type + '_segments_metadata/'
    if not os.path.exists(metadata_dir_name):
        os.makedirs(metadata_dir_name)

    ts_lengths = timeseries_lengths(data_dir)

    # Determine and store the single-month segments table
    one_month_segments = monthly_segments(ts_lengths)
    one_month_filename = metadata_dir_name + '01_months.csv'
    one_month_segments.to_csv(one_month_filename)

    # Determine and store the multi-month segments tables
    for num_months in range(2,18): # Maximum length of the examined timeseries is 18 months
        multi_month_segments = multi_monthly_segments(num_months, one_month_segments)
        multi_month_filename = metadata_dir_name + str(num_months).zfill(2) + '_months.csv'
        multi_month_segments.to_csv(multi_month_filename)


if __name__ == "__main__":
    main()
