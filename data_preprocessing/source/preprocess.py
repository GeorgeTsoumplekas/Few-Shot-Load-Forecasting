"""Preprocessing of the sliced time series'.

First, a number of anomalies and outliers are detected in the original time series. For this,
a number of hyperparameters is determined by parsing the corresponding configuration file, such as
the input data file path, the length of rolling window statistics, the anomaly threshold ratio,
the zero values threshold and the upper and lower iqr bound ratios. Then the detected
anomalies/outliers are interpolated in such a way that the texture and variance of the time series
is preserved.

Typical usage example:
python3 preprocess.py --sliced_dir /path/to/sliced_data/ --config_filepath /path/to/config.yaml
"""

import argparse
import os
from time import time
import yaml

# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def detect_invalid(
    series: pd.Series,
    rolling_window: int,
    anomaly_threshold_ratio: float,
    zero_threshold: float,
    iqr_lower_ratio: float,
    iqr_upper_ratio: float,
    # ts_code: str,
    ) -> pd.DataFrame:
    """
    Detection of the anomalies/outliers in the raw time series.

    The following are detected:
    1) Irregular patterns (anomalies) occuring due to improper interpolation on the raw data
    2) Almost zero values which usually occur from wrong interpolation on the raw data
    3) Extreme outliers

    Args:
        series: The initial unprocessed time series.
        rolling_window: The length of the window used to perform rolling window statistics.
        anomaly_threshold_ratio: A ratio used to define the threshold under which a point is
            considered an anomaly based on its derivative's rolling window std.
        zero_threshold: A threshold under which all values are considered zero and thus invalid.
        iqr_lower_ratio: A ratio defining how many iqrs a point should be further away from the
            25th-percentile value in order to be considered an outlier.
        iqr_upper_ratio: A ratio defining how many iqrs a point should be further away from the
            75th-percentile value in order to be considered an outlier.

    Returns:
        A pandas DataFrame object which contains the original time series along with additional
        calendar features and a column that defines whether the corresponding point is normal or
        an anomaly/outlier.
    """

    # Needed to compute rolling window statistics later
    series_diff = series.diff()

    # Dataframe version of the first differences time series enriched with additional features
    series_enriched = series.reset_index()

    # 1 if point is normal, -1 if it is an outlier/anomaly
    # At first all points are considered normal
    series_enriched['Anomaly'] = np.ones(shape=[series_enriched.shape[0],], dtype=int)

    # Add calendar features to time series
    series_enriched['Date_Time'] = pd.to_datetime(series_enriched['DateTime'])  # Temporary
    series_enriched['Weekday'] = series_enriched.Date_Time.dt.weekday
    series_enriched['Hour'] = [str(val).split()[1] for val in series_enriched['DateTime']]
    series_enriched.drop('Date_Time', axis=1, inplace=True)


    # ANOMALIES DUE TO WRONG INTERPOLATION IN THE ORIGINAL TIME SERIES
    # These anomalies are usually defined by patterns with slow-changing slopes. As a result,
    # when computing the rolling window std of the differences time series, these values will be
    # approximately zero.

    rol_std_series_diff = series_diff['Measurements'].rolling(rolling_window).std()

    anomaly_threshold = anomaly_threshold_ratio * rol_std_series_diff.mean()

    # Shift by the rolling window to detect the beginning of the patterns
    shifted_rol_std_series_diff = rol_std_series_diff.shift(-rolling_window)

    # Fill NaNs with zero and then add the series that capture the heads and tails of the
    # patterns. We don't care for the actual values, just that they are non-zero.
    detected_anomalies_tail = rol_std_series_diff.where(
        rol_std_series_diff < anomaly_threshold).fillna(value=0)
    detected_anomalies_head = shifted_rol_std_series_diff.where(
        shifted_rol_std_series_diff < anomaly_threshold).fillna(value=0)
    detected_anomalies = detected_anomalies_head.add(detected_anomalies_tail)
    detected_anomalies_idx = detected_anomalies[detected_anomalies != 0].index.to_list()

    series_enriched.loc[series_enriched['DateTime'].isin(detected_anomalies_idx),
                        'Anomaly'] = int(-1)

    # # Necessary only for the following plot
    # rol_std_series_diff = rol_std_series_diff.reset_index()
    # rol_std_series_diff['DateTime'] = pd.to_datetime(rol_std_series_diff['DateTime'])
    # rol_std_series_diff = rol_std_series_diff.set_index('DateTime')

    # series_enriched['DateTime'] = pd.to_datetime(series_enriched['DateTime'])
    # series_enriched = series_enriched.set_index('DateTime')

    # _, ax = plt.subplots()
    # a = series_enriched[['Measurements', 'Anomaly']]
    # a.loc[series_enriched['Anomaly'] == 1, 'Measurements'] = np.NaN #anomaly

    # b = series_enriched[['Measurements', 'Anomaly']]
    # b.loc[series_enriched['Anomaly'] == -1, 'Measurements'] = np.NaN

    # ax.plot(b['Measurements'], color='blue', label='Normal', alpha=0.7)
    # ax.plot(a['Measurements'], color='red', label='Anomaly', alpha=0.9)
    # ax.plot(rol_std_series_diff,
    #         color='orange',
    #         label='Second differences rolling std',
    #         alpha=0.7)
    # # plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold', alpha=0.7)
    # plt.legend()
    # plt.title(f"Detection of outlier subsequences for timeseries {ts_code}")
    # plt.xticks(rotation=15)
    # plt.xlabel("Datetime")
    # plt.ylabel("Active power (W)")
    # plt.show()


    # ALMOST ZERO VALUES DUE TO WRONG INTERPOLATION IN THE ORIGINAL TIME SERIES
    # Almost zero values are considered outliers since they usually occur from differentiation
    # of improperly interpolated segments in the original time series.

    series_enriched.loc[(series_enriched['Measurements'] <= zero_threshold) &
                        (series_enriched['Anomaly'] == 1 ),
                        'Anomaly'] = int(-1)

    # # Necessary only for the following plot
    # zero_outliers = series_enriched[series_enriched['Anomaly'] == -1]
    # zero_outliers = zero_outliers['Measurements'].values.tolist()
    # zero_measurements = series_enriched.loc[series_enriched['Measurements'].isin(zero_outliers),
    #                                         ['DateTime', 'Measurements']]
    # zero_measurements['DateTime'] = pd.to_datetime(zero_measurements['DateTime'])

    # series_enriched['DateTime'] = pd.to_datetime(series_enriched['DateTime'])
    # series_enriched = series_enriched.set_index('DateTime')

    # _, ax = plt.subplots()
    # ax.plot(series_enriched['Measurements'], color='blue', label = 'Normal', alpha=0.7)
    # ax.scatter(zero_measurements['DateTime'],
    #            zero_measurements['Measurements'],
    #            color='red',
    #            label = 'Outlier',
    #            marker='x',
    #            alpha=0.9)
    # plt.axhline(y=zero_threshold, color='black', linestyle='--', label='Threshold', alpha=0.7)
    # plt.legend()
    # plt.title(f"Detection of small valued outliers for timeseries {ts_code}")
    # plt.xticks(rotation=15)
    # plt.xlabel("Datetime")
    # plt.ylabel("Active power (W)")
    # plt.show()


    # FIND EXTREME OUTLIERS BASED ON THE IQR OF THE VALID POINTS' DISTRIBUTION
    # Due to the fact that the distribution is highly skewed to the right it makes sense to use
    # a small IQR ratio on the left side to capture any remaining almost zero values and a big IQR
    # ratio on the right side to remove only the the extreme outliers lying on the very end of the
    # distribution's tail.

    # Create a temporary series that contains only the valid points
    # to help us with calculations later
    valid_series_idx = series_enriched.loc[series_enriched['Measurements'] != -1,
                                                           'DateTime'].values.tolist()
    valid_series = series_enriched.loc[series_enriched['DateTime'].isin(valid_series_idx),
                                       ['DateTime', 'Measurements']]

    # Outlier detection based on IQR
    q_1 = valid_series['Measurements'].quantile(0.25)
    q_3 = valid_series['Measurements'].quantile(0.75)
    iqr =  q_3 - q_1
    lower_bound = q_1 - iqr_lower_ratio*iqr
    upper_bound = q_3 + iqr_upper_ratio*iqr

    # Datetimes of measurements that are out of defined bounds
    iqr_outliers_idx = valid_series[(valid_series['Measurements']<lower_bound) |
                                    (valid_series['Measurements']>upper_bound)]
    iqr_outliers_idx = iqr_outliers_idx['DateTime'].values.tolist()

    series_enriched.loc[(series_enriched['DateTime'].isin(iqr_outliers_idx)) &
                        (series_enriched['Anomaly'] == 1 ),
                        'Anomaly'] = int(-1)

    # # Necessary only for the following plots
    # outliers = series_enriched[series_enriched['Anomaly'] == -1]
    # outliers = outliers['Measurements'].values.tolist()
    # measurements = series_enriched.loc[series_enriched['Measurements'].isin(outliers),
    #                                    ['DateTime', 'Measurements']]
    # measurements['DateTime'] = pd.to_datetime(measurements['DateTime'])

    # series_enriched['DateTime'] = pd.to_datetime(series_enriched['DateTime'])
    # series_enriched = series_enriched.set_index('DateTime')

    # _, ax = plt.subplots()
    # ax.plot(series_enriched['Measurements'], color='blue', label = 'Normal', alpha=0.7)
    # ax.scatter(measurements['DateTime'],
    #            measurements['Measurements'],
    #            color='red',
    #            label = 'Outlier',
    #            marker='x',
    #            alpha=0.9)
    # plt.axhline(y=lower_bound, color='black', linestyle='-.', label='Lower bound', alpha=0.7)
    # plt.axhline(y=upper_bound, color='black', linestyle='--', label='Upper Bound', alpha=0.7)
    # plt.legend()
    # plt.title(f"Detection of extreme outliers for timeseries {ts_code}")
    # plt.xticks(rotation=15)
    # plt.xlabel("Datetime")
    # plt.ylabel("Active power (W)")
    # plt.show()

    # plt.figure()
    # series_enriched['Measurements'].plot.hist(bins=100, alpha=0.8)
    # plt.legend()
    # plt.title(f"Distribution of timeseries {ts_code}")
    # plt.xticks(rotation=15)
    # plt.xlabel("Active power (W)")
    # plt.show()

    return series_enriched


def interpolate(series_enriched: pd.DataFrame) -> None:
    """
    Proper interpolation of anomalies and outliers.

    For each value that has to be interpolated, we collect all the values of the time series
    with the same weekday and hour tags and use the mean of them for the interpolation.

    Args:
        series_enriched: A pandas dataframe that contains the initial raw time series as well as
            additional calendar features and a column which discriminates points to normal and
            anomalies/outliers.
    """

    # Datetime stamps of all detected anomalies/outliers
    all_anomalies_idx = series_enriched.loc[series_enriched['Anomaly'] == -1,
                                            'DateTime'].values.tolist()

    for anomaly_idx in all_anomalies_idx:
        # Weekday and hour of examined anomaly point
        anomaly_weekday = series_enriched.loc[series_enriched['DateTime'] == anomaly_idx,
                                              'Weekday'].values[0]
        anomaly_hour = series_enriched.loc[series_enriched['DateTime'] == anomaly_idx,
                                           'Hour'].values[0]

        # Valid measurements with the same weekday and hour tags
        similar_measurements = series_enriched.loc[
            (series_enriched['Weekday'] == anomaly_weekday) &
            (series_enriched['Hour'] == anomaly_hour) &
            (series_enriched['Anomaly'] == 1),
            'Measurements']

        series_enriched.loc[series_enriched['DateTime'] == anomaly_idx,
                            'Measurements'] = similar_measurements.mean()


def preprocess(
    sliced_dir: str,
    rolling_window: int,
    anomaly_threshold_ratio: float,
    zero_threshold: float,
    iqr_lower_ratio: float,
    iqr_upper_ratio: float
    ) -> None:
    """
    Preprocessing of the sliced time series dataset and creation of the preprocessed dataset.

    Initially, each time series in the sliced series dataset is examined for anomalies/outliers.
    Then, they are interpolated and a new preprocessed dataset is created.

    Args:
        sliced_dir: Path to the directory that contains the sliced time series'.
        rolling_window: The length of the window used to perform rolling window statistics.
        anomaly_threshold_ratio: A ratio used to define the threshold under which a point is
            considered an anomaly based on its derivative's rolling window std.
        zero_threshold: A threshold under which all values are considered zero and thus invalid.
        iqr_lower_ratio: A ratio defining how many iqrs a point should be further away from the
            25th-percentile value in order to be considered an outlier.
        iqr_upper_ratio: A ratio defining how many iqrs a point should be further away from the
            75th-percentile value in order to be considered an outlier.
    """

    # Create output directory for preprocessed time series
    preprocessed_dir_name = sliced_dir[:-1] + '_preprocessed/'
    if not os.path.exists(preprocessed_dir_name):
        os.makedirs(preprocessed_dir_name)

    for root, _, files in os.walk(sliced_dir, topdown=False):
        for file in files:

            # LOAD AND CREATE NEEDED DATA STRUCTURES

            file_path = root + '/' + file
            series = pd.read_csv(file_path)
            # series['DateTime'] = pd.to_datetime(series['DateTime'])
            series = series.set_index('DateTime')

            # When the series comes from the initial part of the original time series,
            # then the first value is usuall NaN. As a result, it's better to always remove
            # the first value.
            series = series.iloc[1:]

            # # Plot sliced timeseries
            # ts_code = file[:-4]

            # plt.rcParams["figure.figsize"] = (10, 8)
            # plt.rc('axes', titlesize=14)     # fontsize of the axes title
            # plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
            # plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
            # plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
            # plt.rc('legend', fontsize=10)    # legend fontsize
            # plt.rc('figure', titlesize=14)  # fontsize of the figure title

            # plt.figure()
            # plt.plot(series, color='b', alpha=0.8, label='Sliced timeseries')
            # plt.title(f"Sliced time series {ts_code} before preprocessing")
            # plt.xlabel("Datetime")
            # plt.ylabel("Active power (W)")
            # plt.xticks(rotation=15)
            # plt.show()

            # Detect anomalies/outliers
            series_enriched = detect_invalid(series,
                                             rolling_window,
                                             anomaly_threshold_ratio,
                                             zero_threshold,
                                             iqr_lower_ratio,
                                             iqr_upper_ratio)

            # old_series = series_enriched.loc[:, ['DateTime', 'Measurements']]

            # Proper time series interpolation
            interpolate(series_enriched)

            # old_series['DateTime'] = pd.to_datetime(old_series['DateTime'])
            # old_series = old_series.set_index('DateTime')

            # series_after = series_enriched.loc[:, ['DateTime', 'Measurements']]
            # series_after['DateTime'] = pd.to_datetime(series_after['DateTime'])
            # series_after = series_after.set_index('DateTime')

            # _, ax = plt.subplots()
            # ax.plot(old_series['Measurements'],
            #         color='red',
            #         alpha=0.7,
            #         label='Before preprocessing')
            # ax.plot(series_after['Measurements'],
            #         color='blue',
            #         alpha=0.7,
            #         label='After preprocessing')
            # plt.title(f"Time series {ts_code} before and after preprocessing")
            # plt.xlabel("Datetime")
            # plt.ylabel("Active power (W)")
            # plt.xticks(rotation=15)
            # plt.legend()
            # plt.show()

            # Output preprocessed time series to a .csv file
            preprocessed_filename = preprocessed_dir_name + file
            series_enriched.loc[:, ['DateTime', 'Measurements']].to_csv(preprocessed_filename,
                                                                        index=False)

            # input("Press enter to continue...")


def main() -> None:
    """
    Initially, parsing the command line arguments that define the hyperparameters' values
    of the anomalies/outliers detection process and then preprocessing the desired time series'
    based on the steps described above.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--sliced_dir', '-D', dest='sliced_dir')
    parser.add_argument('--config', '-C', dest='config_filepath')
    args = parser.parse_args()

    config_filepath = args.config_filepath
    with open(config_filepath, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    sliced_dir = args.sliced_dir
    rolling_window = config['rolling_window']
    anomaly_threshold_ratio = config['anomaly_threshold_ratio']
    zero_threshold = config['zero_threshold']
    iqr_lower_ratio = config['iqr_lower_ratio']
    iqr_upper_ratio = config['iqr_upper_ratio']

    print('Preprocessing sliced time series\'...')
    start = time()
    preprocess(sliced_dir,
               rolling_window,
               anomaly_threshold_ratio,
               zero_threshold,
               iqr_lower_ratio,
               iqr_upper_ratio)
    end = time()

    print('Preprocessing completed.')
    print(f"Elapsed time: {end-start:.2f}s")


if __name__ == "__main__":
    main()
