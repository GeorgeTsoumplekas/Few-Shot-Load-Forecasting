"""Slicing of the raw time series'.

At first, the time series' first and second differences are calculated. Extreme outliers are removed
from the first differences time series. Based on the values of the second differences time series,
the first differences time series is sliced and only its useful parts are kept. Parts of the
time series that are flat or irregular (probably due to errors) are removed. Finally, the useful
parts are saved as .csv files in a separate directory.

Typical usage example:
python3 slice.py --raw_dir /path/to/raw_data/ --config /path/to/config.yaml
"""

import argparse
import os
from time import time
import warnings
import yaml

# from matplotlib import pyplot as plt
import pandas as pd


# Ignore unimportant user warnings that appear
warnings.filterwarnings("ignore")


def create_first_differences(
    filename: str,
    ) -> pd.DataFrame:
    """Extract first differences time series from the raw data.

    Raw data is loaded and used to calculate the first differences time series. Only the date time
    and the measurement values are kept.

    Args:
        filename: A str that contains the path to the raw time series.
    Returns:
        A pandas DataFrame that contains the first differences time series.
    """

    data = pd.read_csv(filename, header=None, names=['DateTime', 'Measurements'])

    # Transform date time to a format readable by humans
    data['DateTime'] = pd.to_datetime(data['DateTime'], unit='ms')
    data = data.set_index('DateTime')

    # Multiplier used to transform to measurements to active power levels
    mult_factor = int(60 / ((data.index[1] - data.index[0]).seconds // 60) % 60)

    first_diff = data.diff()
    first_diff['Measurements'] = mult_factor*first_diff['Measurements']

    return first_diff


def create_second_differences(
    first_diff: pd.DataFrame,
    ) -> pd.DataFrame:
    """Extract second differences time series form the first differences time series.

    The second differences time series is calculated as well as a shifted by one to the left
    version of it. The NaN values are also filled with zeros. FInally, the absolute values of the
    second differences and the shifted version are returned. These will be necessary to find the
    useful parts of the first differences time series later.

    Args:
        first_diff: A pandas DataFrame that contains the first differences time series.
    Returns:
        A pandas DataFrame that contains the second differences time series and its shifted version.
    """

    sec_diff = first_diff.diff()

    # Create shifted column
    sec_diff['Next'] = sec_diff.Measurements.shift(-1)

    # Fill NaNs
    sec_diff.fillna(0, inplace=True)

    # Transform to absolute values
    sec_diff['Measurements'] = sec_diff['Measurements'].abs()
    sec_diff['Next'] = sec_diff['Next'].abs()

    return sec_diff


def remove_outliers(
    first_diff: pd.DataFrame,
    outlier_threshold: float,
    ) -> None:
    """Remove extreme outliers from the first differences time series.

    This includes values that exceed an outlier threshold and are replaced with zero.

    Args:
        first_diff: A pandas DataFrame that contains the first differences time series.
        outlier_threshold: A float above which all values are considered outliers.
    """

    # Find outliers
    outliers = first_diff.loc[first_diff['Measurements'].abs() > outlier_threshold]

    # Replace with 0
    for index in outliers.index:
        first_diff['Measurements'].loc[index] = 0


def find_crosspoints(
    sec_diff: pd.DataFrame,
    cross_threshold: float,
    ) -> pd.DataFrame:
    """Find the transitions from flat parts to useful parts of time series and vice versa.

    A transition or cross is defined as the datetime for which its value and its next one are not
    both above or below a user-defined threshold. A positive cross is when a value is below the
    threshold and its next one is above, the opposite being a negative cross. Both positive
    and negative crosses are found and merged together before being ordered based on their datetime.

    Args:
        sec_diff: A pandas DataFrame that contains the second differences time series and its
            shifted version.
        cross_threshold: A float based on which the crosses are defined.
    Returns:
        A pandas DataFrame that contains all crosses sorted based on their datetime.
    """

    # True if point corresponds to a positive cross
    sec_diff['pos_cross'] = ((sec_diff.Measurements <= cross_threshold) &
                             (sec_diff.Next > cross_threshold))

    # True if point corresponds to a negative cross
    sec_diff['neg_cross'] = ((sec_diff.Measurements > cross_threshold) &
                             (sec_diff.Next <= cross_threshold))

    # Dataframe with positive crosses only
    positive_crosses = sec_diff.loc[sec_diff['pos_cross']]
    positive_crosses.reset_index(inplace=True)
    positive_crosses = positive_crosses.drop(['Measurements', 'Next', 'pos_cross', 'neg_cross'],
                                             axis=1)
    positive_crosses['Sign'] = 1

    # DataFrame with negative crosses only
    negative_crosses = sec_diff.loc[sec_diff['neg_cross']]
    negative_crosses.reset_index(inplace=True)
    negative_crosses = negative_crosses.drop(['Measurements', 'Next', 'pos_cross', 'neg_cross'],
                                             axis=1)
    negative_crosses['Sign'] = -1

    # Merge crosses and sort based on their datetime
    crosses = pd.merge_ordered(positive_crosses, negative_crosses, on=['DateTime', 'Sign'])

    return crosses


def find_useful_parts(
    crosses: pd.DataFrame,
    sec_diff: pd.DataFrame,
    min_length: str,
    acceptable_length: str,
    ) -> pd.DataFrame:
    """Find the useful parts of the time series' based on their length and values and save them.

    A three pointers technique is used to find the useful parts. A part is considered useful when
    it's not flat and its length is long enough. Flat parts are generally considered non useful
    unless they have a short length and preceed or follow useful parts. Then they can be considered
    as useful and be interpolated later.

    Args:
        crosses: A pandas DataFrame that contains the datetime and type of the various crosses.
        sec_diff: A pandas DataFrame that contains the second differences time series and its
            shifted version.
        min_length: A string that contains the length above which a flat part is considered not
            useful.
        acceptable_length: A string that contains the length above which a non-flat part is
            considered useful.
    Returns:
        A pandas DataFrame that contains the start and end datetimes of the useful parts of a time
        series.
    """

    chunks = pd.DataFrame(columns=['start', 'end'])

    # Start pointers
    ptr_1 = crosses['DateTime'].iloc[0]
    ptr_2 = crosses['DateTime'].iloc[0]

    end = sec_diff.index[-1]

    # Counter of useful parts in a time series
    counter = 0

    # While we have not reached the end of the time series
    while (ptr_1 < end) & (ptr_2 < end):
        # Point to the next cross or the end if there are not any other crosses after
        # the start pointers
        ptr_3 = crosses[crosses['DateTime'] >= ptr_2][crosses['Sign'] == 1]['DateTime']
        if len(ptr_3.index) == 0:
            ptr_3 = end
        else:
            ptr_3 = ptr_3.iloc[0]

        # If the flat part is small and can be interpolated include it in the useful part
        if (ptr_3-ptr_2) < min_length:
            # Examine after the small flat part
            ptr_2 = crosses[crosses['DateTime'] > ptr_3][crosses['Sign'] == -1]['DateTime']
            if len(ptr_2.index) == 0:
                ptr_2 = end
                # If the non-flat part is long enough, save it
                if (ptr_2-ptr_1) > acceptable_length:
                    chunks = chunks.append(pd.DataFrame({'start': ptr_1, 'end': ptr_2},
                                                        index=[counter]))
                    counter += 1
            else:
                ptr_2 = ptr_2.iloc[0]
        else:
            # If the non-flat part is long enough, save it
            if (ptr_2-ptr_1) > acceptable_length:
                chunks = chunks.append(pd.DataFrame({'start': ptr_1, 'end': ptr_2},
                index=[counter]))
                counter += 1
            # Move on to the next part
            ptr_1 = ptr_3
            ptr_2 = crosses[crosses['DateTime'] > ptr_3][crosses['Sign'] == -1]['DateTime']
            if len(ptr_2.index) == 0:
                ptr_2 = end
            else:
                ptr_2 = ptr_2.iloc[0]

    return chunks


def save_useful_parts(
    chunks: pd.DataFrame,
    sliced_dir_name: str,
    timeseries_code: str,
    first_diff: pd.DataFrame,
    ) -> None:
    """Save the useful parts of the first differences time series in a new directory.

    The parts are saved as .csv files.

    Args:
        chunks: A pandas DataFrame that contains the start and end datetimes of the useful parts
        of a time series.
        sliced_dir_name: A string that contains the name of the directory on which the sliced
            time series' will be saved.
        timeseries_code: A string that is the id of the examined timeseries.
        first_diff: A pandas DataFrame that contains the first differences time series.
    """

    for i, (_, date) in enumerate(chunks.iterrows()):
        target_file = sliced_dir_name + timeseries_code + str(i+1) + '.csv'
        useful_part = first_diff['Measurements'][date['start']:date['end']]
        useful_part.to_csv(target_file)


def slicing(
    raw_dir: str,
    cross_threshold: float,
    min_length: str,
    acceptable_length: str,
    outlier_threshold: float,
    ) -> None:
    """Slicing of the raw time series dataset and creation of the sliced dataset.

    Initially, the first and second differences time series' are calculated and the extreme
    outliers are removed. The crosses are calculated based on which the useful parts of the
    time series are found and finally saved.

    Args:
        raw_dir: Path to the directory that contains the raw time series'.
        cross_threshold: A float based on which the crosses are defined.
        min_length: A string that contains the length above which a flat part is considered not
            useful.
        acceptable_length: A string that contains the length above which a non-flat part is
            considered useful.
        outlier_threshold: A float above which all values are considered outliers.
    """

    # Create output directory for sliced time series'
    sliced_dir_name = raw_dir[:-1] + '_sliced/'
    if not os.path.exists(sliced_dir_name):
        os.makedirs(sliced_dir_name)

    # # Plot parameters
    # plt.rcParams["figure.figsize"] = (10,8)
    # plt.rc('axes', titlesize=14)     # fontsize of the axes title
    # plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=10)    # legend fontsize
    # plt.rc('figure', titlesize=14)  # fontsize of the figure title

    # # Plots directory
    # plots_target_dir = './outlier_plots/'
    # if not os.path.exists(plots_target_dir):
    #     os.makedirs(plots_target_dir)

    for root, _, files in os.walk(raw_dir, topdown=False):
        for i, file in enumerate(files):
            filename = root + file
            timeseries_code = filename[-45:-25]

            first_diff = create_first_differences(filename)

            # plt.figure(figsize=(10, 8))

            # fig, axs = plt.subplots(2)
            # axs[0].plot(first_diff, color='r', alpha=0.7, label='Before removing outliers')

            remove_outliers(first_diff, outlier_threshold)

            # plt.plot(first_diff, color='b', alpha=0.8, label='First differences')

            # axs[1].plot(first_diff, color='b', alpha=0.7, label='After removing outliers')

            # # axs[0].set(xlabel='Datetime')
            # axs[1].set(xlabel='Datetime')

            # axs[0].set(ylabel='Power (W)')
            # axs[1].set(ylabel='Power (W)')

            # axs[0].legend()
            # axs[1].legend()

            # fig.suptitle(f"Outliers in timeseries {timeseries_code[:-1]}")
            # plt.legend()
            # plt.show()

            # # plots_target_filename = plots_target_dir + timeseries_code[:-1] + '.png'
            # # plt.savefig(plots_target_filename, dpi=150)

            sec_diff = create_second_differences(first_diff)

            # plt.plot(sec_diff['Measurements'], color='r', alpha=0.6, label='Second differences')
            # plt.xticks(rotation=30)
            # plt.xlabel('Datetime')
            # plt.ylabel('Power (W)')
            # plt.title(f"Time series {timeseries_code[:-1]}")
            # plt.legend()
            # plt.show()

            crosses = find_crosspoints(sec_diff, cross_threshold)

            # If no crosses exist this means that either the whole series is useful or useless.
            # For simplicity reasons it's just easier to skip such time series, since they are
            # rare and do not immensely affect the results.
            if len(crosses) == 0:
                continue

            chunks = find_useful_parts(crosses, sec_diff, min_length, acceptable_length)

            # plt.figure(figsize=(10, 8))
            # plt.plot(first_diff, color='r', alpha=0.7, label='Discarded part')
            # for i, (_, date) in enumerate(chunks.iterrows()):
            #     useful_part = first_diff['Measurements'][date['start']:date['end']]
            #     plt.plot(useful_part, color='b', alpha=0.9-i*0.25, label=f"Useful part {i+1}")
            # plt.xticks(rotation=30)
            # plt.xlabel('Datetime')
            # plt.ylabel('Power (W)')
            # plt.title(f"Slicing of timeseries {timeseries_code[:-1]}")
            # plt.legend()
            # plt.show()

            # plots_target_filename = plots_target_dir + timeseries_code[:-1] + '.png'
            # plt.savefig(plots_target_filename, dpi=150)

            save_useful_parts(chunks, sliced_dir_name, timeseries_code, first_diff)

            print(f"{i+1}|{len(files)}")


def main() -> None:
    """
    Initially, parsing the command line arguments that define the hyperparameters' values
    of the slicing process and then slicing the desired time series' based on the steps
    described above.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', '-D', dest='raw_dir')
    parser.add_argument('--config', '-C', dest='config_filepath')
    args = parser.parse_args()

    config_filepath = args.config_filepath
    with open(config_filepath, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    raw_dir = args.raw_dir
    cross_threshold = config['cross_threshold']
    min_length = pd.Timedelta(config['min_length'])
    acceptable_length = pd.Timedelta(config['acceptable_length'])
    outlier_threshold = config['outlier_threshold']

    print('Slicing raw time series\'...')
    start = time()
    slicing(raw_dir,
          cross_threshold,
          min_length,
          acceptable_length,
          outlier_threshold)
    end = time()

    print('Slicing completed.')
    print(f"Elapsed time: {end-start:.2f}s")


if __name__ == "__main__":
    main()
