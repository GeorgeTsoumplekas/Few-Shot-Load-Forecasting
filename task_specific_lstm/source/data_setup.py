"""Module that contains classes and functions necessary for data preprocessing.

More specifically, the functionality of the code included here is to ingest raw time series data
and transform them in the appropriate format to be used by a PyTorch model. This includes spliting
data to train and test sets and creating the corresponding DataLoaders. Additionally, a custom
Dataset class is defined.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    """A custom PyTorch Dataset class to be subsequnetly used by the DataLoaders.

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


def build_dataset(x_train, y_train, x_test, y_test, batch_size):
    """Instantiate DataLoaders for the training and test sets.

    Args:
        x_train: A torch.Tensor that contains the input features of the training set.
        y_train: A torch.Tensor that contains the output values of the training set.
        x_test: A torch Tensor that contains the input features of the test set.
        y_test: A torch.Tensor that contains the output values of the test set.
        batch_size: An integer that defines the number of samples in each batch.
    Returns:
        Two torch DataLoader objects, one for the training and one for the test set.
    """

    # Instantiate Datasets
    train_data = MyDataset(x_data=x_train,
                           y_data=y_train)
    test_data = MyDataset(x_data=x_test,
                          y_data=y_test)

    # Instantiate DataLoaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=os.cpu_count(),  # Use all cpus available
                                  shuffle=False)
    test_dataloader = DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  num_workers=os.cpu_count(),  # Use all cpus available
                                  shuffle=False)

    return train_dataloader, test_dataloader


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


def split_train_test(data, data_split_constants):
    """Ingest the raw time series and create the desired train and test sets.

    First, both input and output length are defined, as well as the size of the test set.
    Based on that test size, the last elements of the dataset become the test set and the rest
    the training set. Then both of these sets are split into subsequences: the subsequences of
    the training set are non-overlapping, while the subsequences on the test set are overlapping
    based on rolling window that each time rolls by the size of the output subsequence (see the
    report for more details and diagrams of the process). Finally, the subsequences are normalized
    and transformed into torch tensors.

    Args:
        data: A numpy.ndarray that contains the raw time series.
        data_split_constants: A dictionary with values that control correct splitting of the data
            such as the size of the train/test sets and the length of input/output subsequences.
    Returns:
        A tuple that contains the input and output subsequences of the training and test sets as
        torch Tensors.
    """

    timestep = data_split_constants['timestep']  # Minutes between measurements
    week_num = data_split_constants['week_num']  # Number of weeks in the input subsequence
    pred_days = data_split_constants['pred_days']  # Output subsequence length in days
    test_days = data_split_constants['test_days']  # Test set size in days

    # Number of days in each input subsequence
    x_seq_days = week_num*7

    # Number of measurements in each day
    day_measurements = (24*60) // timestep

    # Number of measurements in each input subsequence
    x_seq_measurements = day_measurements*x_seq_days

    # Number of measurements in each output subsequence
    y_seq_measurements = day_measurements*pred_days

    x_train = np.empty((0, x_seq_measurements))
    x_test = np.empty((0, x_seq_measurements))
    y_train = np.empty((0, y_seq_measurements))
    y_test = np.empty((0, y_seq_measurements))

    total_days = len(data)//day_measurements

    # The dataset should be divisible by the number of measurements in a day
    data = data[-total_days*day_measurements:]

    # Number of training set subsequences
    x_train_seqs = (total_days - test_days - pred_days) // x_seq_days

    for i in range(x_train_seqs):
        x_slice = data[i*x_seq_measurements:(i+1)*x_seq_measurements]
        y_slice = data[(i+1)*x_seq_measurements:(i+1)*x_seq_measurements+y_seq_measurements]

        x_train = np.vstack((x_train, x_slice))
        y_train = np.vstack((y_train, y_slice))

    for i in range(test_days):
        x_slice = data[len(data)-x_seq_measurements-y_seq_measurements-(test_days-i)+1:
                        len(data)-y_seq_measurements-(test_days-i)+1]
        y_slice = data[len(data)-y_seq_measurements-(test_days-i)+1:len(data)-(test_days-i)+1]

        x_test = np.vstack((x_test, x_slice))
        y_test = np.vstack((y_test, y_slice))

    # Normalize using the mean and std of the training set (to avoid data leakage)
    train_mean = x_train.reshape((-1,)).mean()
    train_std = x_train.reshape((-1,)).std()

    x_train = (x_train - train_mean) / train_std
    y_train = (y_train - train_mean) / train_std

    x_test = (x_test - train_mean) / train_std
    y_test = (y_test - train_mean) / train_std

    x_train = torch.from_numpy(x_train).to(torch.float32)
    x_test = torch.from_numpy(x_test).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.float32)
    y_test = torch.from_numpy(y_test).to(torch.float32)

    return x_train, y_train, x_test, y_test
