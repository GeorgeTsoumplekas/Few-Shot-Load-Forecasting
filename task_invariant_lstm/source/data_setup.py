"""Module that contains classes and functions necessary for data preprocessing.

This modules contains two different custom torch Datasets: one to handle the sets of tasks and
another one to handle dataset of each task individually. Dedicated methods are included to load,
trim and concatenate all timeseries together as well as create the task set's datasets and
dataloaders. Within each task, the time series is split in input/output subsequences and the
corresponding datasets and dataloaders are created. The process differs for training and
test tasks.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class AcrossTasksDataset(Dataset):
    """A custom PyTorch Dataset to handle the set of tasks.

    The dataset is defined in such a way that each batch corresponds to a single task
    (time series). However it is not necessary for each time series to have the same length.

    Attributes:
        data: A torch.Tensor that contains all time series concatenated together.
        batch_idx: A list of integers where each integer corresponds to the beginning of a
            different time series in the concatenated timeseries array (data).
        day_measurements: An integer for the number of measurements in the span of a single day.
        timeseries_codes: A list of strings where each string is a unique id of a timeseries.
    """
    def __init__(self, data, batch_idx, day_measurements, filenames):
        """Init AcrossTasksDataset with the appropriate attribute values."""

        self.data = data
        self.batch_idx = batch_idx
        self.day_measurements = day_measurements
        self.timeseries_codes = [filename[-7:-4] for filename in filenames]

    def __getitem__(self, index):
        """Get the data of a specific task based on the given index.

        The index is used indirectly; based on this we find the beginning and end index of the
        timeseries in the concatenated timeseries tensor and we retrieve the timeseries based on
        those. The code of the selected time series is returned, too.
        """

        start_idx = self.batch_idx[index]*self.day_measurements
        end_idx = self.batch_idx[index+1]*self.day_measurements
        return self.data[start_idx:end_idx], self.timeseries_codes[index]

    def __len__(self):
        """Get the number of tasks (timeseries) in the tasks set."""
        return len(self.batch_idx) - 1


class TaskSpecificDataset(Dataset):
    """A custom PyTorch Dataset to handle the data within each task.

    The dataset is defined in such a way that each batch corresponds to a single
    subsequence/sample.

    Attributes:
        x_data: A torch.Tensor that contains the input features of the dataset samples.
        y_data: A torch.Tensor that contains the output values of the dataset samples.
    """

    def __init__(self, x_data, y_data):
        """Init MyDataset with input features and corresponding outputs."""

        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, idx):
        """Get a specific input/output sample from the dataset based on its index."""

        x_sample = self.x_data[idx]
        y_sample = self.y_data[idx]
        return x_sample, y_sample

    def __len__(self):
        """Get the number of samples in the dataset."""

        return self.x_data.shape[0]


def load_timeseries(filename):
    """Loads the raw time series from the specified file.

    Args:
        filename: A string which defines the path to the desired time series file.
    Returns:
        A numpy.ndarray that contains the raw time series.
    """
    series = pd.read_csv(filename)
    data = series['Measurements'].to_numpy()

    return data


def trim_timeseries(data, day_measurements, pred_days, week_num, test_days):
    """Trims down the length of the timeseries.

    Only the last measurements are kept, in such a way that the remaining measurements
    match exactly with the length of the input/output subsequences that will be created later.

    Args:
        data: A numpy.ndarray that contains the time series.
        day_measurements: An integer for the number of measurements in a single day.
        pred_days: An integer that corresponds to the output subsequence length in days.
        week_num: An integer that corresponds to the number of weeks in the input sequence.
        test_days: An integer that corresponds to the test set size in days.
    Returns:
        A numpy.ndarray that contains the trimmed time series.
    """

    # Number of days in each input subsequence
    x_seq_days = week_num*7

    # Total number of days in time series
    total_days = len(data)//day_measurements

    # The dataset should be divisible by the number of measurements in a day
    data = data[-total_days*day_measurements:]

    # Remove the remainder days from the beginning of the time series
    remainder_days = (total_days - pred_days - test_days) % x_seq_days
    remainder_measurements = remainder_days*day_measurements
    data = data[remainder_measurements:]

    return data


def concat_tasks(data_filenames, pred_days, test_days, week_num, day_measurements):
    """Concatenate all given time series in a single torch tensor.

    The time series are loaded trimmed and then concatenated together before being transformed
    to a torch tensor. A list that contains the index where each individual time series starts
    within the concatenated tensor is also created.

    Args:
        data_filenames: A list of strings where each string is the path to a time series.
        pred_days: An integer that corresponds to the output subsequence length in days.
        test_days: An integer that corresponds to the test set size in days.
        week_num: An integer that corresponds to the number of weeks in the input sequence.
        day_measurements: An integer for the number of measurements in a single day.
    Returns:
        - A torch.Tensor that contains all the time series concatenated together.
        - A list of integers that contains the length of each timeseries.
    """

    chunk_sizes = [0]
    total_data = torch.Tensor(0)

    for filename in data_filenames:
        data = load_timeseries(filename)

        data = trim_timeseries(data, day_measurements, pred_days, week_num, test_days)

        # The starting indexes are counted in days and not measurements to avoid extremely
        # large index values.
        chunk_sizes.append(int(data.shape[0]/day_measurements))

        data = torch.Tensor(data)
        total_data = torch.cat([total_data, data], dim=0)

    return total_data, chunk_sizes


def get_tasks_dataset(data_filenames, pred_days, test_days, week_num, day_measurements):
    """Create the dataset object that handles the tasks.

    The tasks are concatenated and the list of time series start indexes is created,
    before creating the tasks dataset.

    Args:
        data_filenames: A list of strings where each string is the path to a time series.
        pred_days: An integer that corresponds to the output subsequence length in days.
        test_days: An integer that corresponds to the test set size in days.
        week_num: An integer that corresponds to the number of weeks in the input sequence.
        day_measurements: An integer for the number of measurements in a single day.
    Returns:
        An AcrossTasksDataset object that contains all given tasks.
    """

    tasks_set, chunk_sizes = concat_tasks(data_filenames,
                                          pred_days,
                                          test_days,
                                          week_num,
                                          day_measurements)

    # The start indexes of each timeseries can be calculated as the cumulative sum of the length
    # of the previous time series'.
    task_batch_indexes = np.cumsum(chunk_sizes).tolist()

    tasks_dataset = AcrossTasksDataset(tasks_set,
                                       task_batch_indexes,
                                       day_measurements,
                                       data_filenames)
    return tasks_dataset


def build_tasks_set(data_filenames, data_config, task_batch_size, train_task):
    """Create a dataloader that handles the dataset of the given tasks.

    Args:
        data_filenames: A list of strings where each string is the path to a time series.
        data_config: A dictionary that contains various user-configured values (i.e. train/test
            set sizes and subsequences' length).
        task_batch_size: An integer that represents the number of tasks in each tasks batch.
        train_task: A boolean flag that defines whether the created dataloader refers to a
            dataset of tasks used for training or not.
    Returns:
        A torch dataloader object that contains the given tasks.
    """

    # CONSTANTS
    day_measurements = data_config['day_measurements']
    week_num = data_config['week_num']
    pred_days = data_config['pred_days']
    test_days = None

    # Training tasks do not have a test set
    if train_task is True:
        test_days = 0
    else:
        test_days = data_config['test_days']

    tasks_dataset = get_tasks_dataset(data_filenames,
                                      pred_days,
                                      test_days,
                                      week_num,
                                      day_measurements)

    tasks_dataloader = DataLoader(dataset=tasks_dataset,
                                  batch_size=task_batch_size,
                                  shuffle=True)  # Tasks are not sequential so they can be shuffled

    return tasks_dataloader


def build_train_task(train_task_data, sample_batch_size, data_config):
    """Create the dataloader that handles a specific task used for training.

    The time series that corresponds to the task is initially split to input/output subsequences
    and then the corresponding dataset and dataloader objects are created.

    Args:
        train_task_data: A torch.Tensor that corresponds to the time series of the task.
        sample_batch_size: An integer that is the number of subsequences in each batch.
        data_config: A dictionary that contains various user-configured values.
    Returns:
        A dataloader that handles the samples of the training task.
    """

    train_task_data = train_task_data.squeeze()

    # Only training subsequences used in training tasks
    x_task_train, y_task_train = split_train_task(train_task_data, data_config)

    train_dataset = TaskSpecificDataset(x_task_train, y_task_train)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=sample_batch_size,
                                  num_workers=os.cpu_count(),  # Use all cpus available
                                  shuffle=False)  # Sequential data, so no shuffling

    return train_dataloader


def build_test_task(test_task_data, sample_batch_size, data_config):
    """Create the dataloader that handles a specific task used for testing.

    The time series that corresponds to the task is initially split to input/output subsequences
    (both for the training and test sets) and then the corresponding dataset and dataloader
    objects are created.

    Args:
        test_task_data: A torch.Tensor that corresponds to the time series of the task.
        sample_batch_size: An integer that is the number of subsequences in each batch.
        data_config: A dictionary that contains various user-configured values.
    Returns:
        Two torch DataLoader objects, one for the training and one for the test set of the task.
    """

    test_task_data = test_task_data.squeeze()

    # Both training and test subsequences are created in test tasks.
    x_task_train, y_task_train, x_task_test, y_task_test = split_test_task(test_task_data,
                                                                           data_config)

    train_dataset = TaskSpecificDataset(x_task_train, y_task_train)
    test_dataset = TaskSpecificDataset(x_task_test, y_task_test)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=sample_batch_size,
                                  num_workers=os.cpu_count(),  # Use all cpus available
                                  shuffle=False)  # Sequential data, so no shuffling
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=sample_batch_size,
                                 num_workers=os.cpu_count(),  # Use all cpus available
                                 shuffle=False)  # Sequential data, so no shuffling

    return train_dataloader, test_dataloader


def split_train_task(task_data, data_config):
    """Ingest the raw time series and create the desired training set.

    First, both input and output length are definedand based on that the number of training
    subsequences is calculated. Then the whole time series is split into non-overlapping
    subsequences. Finally, the subsequences are normalized and transformed into torch tensors.
    Note that there is no test set in a training task so all time series is used for training.

    Args:
        task_data: A torch.Tensor that corresponds to the time series of the task.
        data_config: A dictionary that contains various user-configured values.
    Returns:
        A tuple that contains the input and output subsequences of the training set as
        torch Tensors.
    """

    day_measurements = data_config['day_measurements']
    week_num = data_config['week_num']
    pred_days = data_config['pred_days']

    # Number of days in each input subsequence
    x_seq_days = week_num*7

    # Number of measurements in each input subsequence
    x_seq_measurements = day_measurements*x_seq_days

    # Number of measurements in each output subsequence
    y_seq_measurements = day_measurements*pred_days

    x_task_train = torch.empty((0, x_seq_measurements))
    y_task_train = torch.empty((0, y_seq_measurements))

    total_days = len(task_data)//day_measurements

    # Number of training set subsequences
    x_train_seqs = (total_days - pred_days) // x_seq_days

    # Normalize training set
    task_data_mean = torch.mean(task_data)
    task_data_std = torch.std(task_data)
    task_data = (task_data - task_data_mean) / task_data_std

    for i in range(x_train_seqs):
        x_slice = task_data[i*x_seq_measurements:(i+1)*x_seq_measurements]
        y_slice = task_data[(i+1)*x_seq_measurements:(i+1)*x_seq_measurements+y_seq_measurements]

        x_task_train = torch.vstack((x_task_train, x_slice))
        y_task_train = torch.vstack((y_task_train, y_slice))

    return x_task_train, y_task_train


def split_test_task(task_data, data_config):
    """Ingest the raw time series and create the desired train and test sets.

    First, both input and output length are defined, as well as the size of the test set.
    Based on that test size, the last elements of the dataset become the test set and the rest
    the training set. Then both of these sets are split into subsequences: the subsequences of
    the training set are non-overlapping, while the subsequences on the test set are overlapping
    based on rolling window that each time rolls by the size of the output subsequence (see the
    report for more details and diagrams of the process). Finally, the subsequences are normalized
    and transformed into torch tensors.

    Args:
        task_data: A torch.Tensor that contains the time series of the task.
        data_config: A dictionary that contains various user-configured values.
    Returns:
        A tuple that contains the input and output subsequences of the training and test sets as
        torch Tensors.
    """

    day_measurements = data_config['day_measurements']
    week_num = data_config['week_num']
    pred_days = data_config['pred_days']
    test_days = data_config['test_days']

    # Number of days in each input subsequence
    x_seq_days = week_num*7

    # Number of measurements in each input subsequence
    x_seq_measurements = day_measurements*x_seq_days

    # Number of measurements in each output subsequence
    y_seq_measurements = day_measurements*pred_days

    x_task_train = torch.empty((0, x_seq_measurements))
    x_task_test = torch.empty((0, x_seq_measurements))
    y_task_train = torch.empty((0, y_seq_measurements))
    y_task_test = torch.empty((0, y_seq_measurements))

    total_days = len(task_data)//day_measurements

    # The dataset should be divisible by the number of measurements in a day
    task_data = task_data[-total_days*day_measurements:]

    # Number of training set subsequences
    x_train_seqs = (total_days - test_days - pred_days) // x_seq_days

    for i in range(x_train_seqs):
        x_slice = task_data[i*x_seq_measurements:(i+1)*x_seq_measurements]
        y_slice = task_data[(i+1)*x_seq_measurements:(i+1)*x_seq_measurements+y_seq_measurements]

        x_task_train = torch.vstack((x_task_train, x_slice))
        y_task_train = torch.vstack((y_task_train, y_slice))

    for i in range(test_days):
        x_slice = task_data[len(task_data)-x_seq_measurements-y_seq_measurements-(test_days-i)+1:
                        len(task_data)-y_seq_measurements-(test_days-i)+1]
        y_slice = task_data[len(task_data)-y_seq_measurements-(test_days-i)+1:
                            len(task_data)-(test_days-i)+1]

        x_task_test = torch.vstack((x_task_test, x_slice))
        y_task_test = torch.vstack((y_task_test, y_slice))

    # Normalize using the mean and std of the training set (to avoid data leakage)
    train_mean = x_task_train.reshape((-1,)).mean()
    train_std = x_task_train.reshape((-1,)).std()

    x_task_train = (x_task_train - train_mean) / train_std
    y_task_train = (y_task_train - train_mean) / train_std

    x_task_test = (x_task_test - train_mean) / train_std
    y_task_test = (y_task_test - train_mean) / train_std

    return x_task_train, y_task_train, x_task_test, y_task_test
