"""Module that contains classes and functions necessary for data preprocessing.

More specifically, the functionality of the code included here is to ingest raw time series data
and transform them in the appropriate format to be used by a PyTorch model. This includes spliting
data to train and test sets and creating the corresponding DataLoaders. Additionally, a custom
Dataset class is defined.
"""

import os

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
        A torch tensor that contains the raw time series.
    """

    series = pd.read_csv(filename)
    data = torch.tensor(series['Measurements'].to_numpy(), dtype=torch.float32)

    return data


def save_normalization_settings(y_min, y_max, target_dir_name):
    """Save variables used in transforming the raw data as a .csv file.

    This is necessary for re-transforming the data back to its original scale later.

    Args:
        y_min: A torch tensor that contains the minimum value of the output subsequneces.
        y_max: A torch tensor that contains the maximum value of the output subsequences.
        target_dir_name: A string with the name of the directory the results will be saved.
    """

    standardization_settings = pd.DataFrame({
        'y_min': [y_min.item()],
        'y_max': [y_max.item()],
    })

    filepath = target_dir_name + 'settings.csv'
    standardization_settings.to_csv(filepath, index=False)


def split_train_test(data, data_split_constants, results_dir_name):
    """Ingest the raw time series and create the desired train and test sets.

    First, both input and output length are defined, as well as the size of the test set.
    Based on that test size, the last elements of the dataset become the test set and the rest
    the training set. Then both of these sets are split into subsequences: the subsequences of
    the training set are non-overlapping, while the subsequences on the test set are overlapping
    based on rolling window that each time rolls by the size of the output subsequence (see the
    report for more details and diagrams of the process). Finally, the subsequences are normalized
    and shifted to be positive.

    Args:
        data: A torch tensor that contains the raw time series.
        data_split_constants: A dictionary with values that control correct splitting of the data
            such as the size of the train/test sets and the length of input/output subsequences.
        results_dir_name: A string with the name of the directory the results will be saved.
    Returns:
        A tuple that contains the input and output subsequences of the training and test sets as
        torch Tensors as well as as the output subsequences of the test set unstandardized.
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

    x_train = torch.empty((0, x_seq_measurements))
    x_test = torch.empty((0, x_seq_measurements))
    y_train = torch.empty((0, y_seq_measurements))
    y_test = torch.empty((0, y_seq_measurements))

    total_days = len(data)//day_measurements

    # The dataset should be divisible by the number of measurements in a day
    data = data[-total_days*day_measurements:]

    # Number of training set subsequences
    x_train_seqs = (total_days - test_days - pred_days) // x_seq_days

    for i in range(x_train_seqs):
        x_slice = data[i*x_seq_measurements:(i+1)*x_seq_measurements]
        y_slice = data[(i+1)*x_seq_measurements:(i+1)*x_seq_measurements+y_seq_measurements]

        x_train = torch.vstack((x_train, x_slice))
        y_train = torch.vstack((y_train, y_slice))

    for i in range(test_days):
        x_slice = data[len(data)-x_seq_measurements-y_seq_measurements-(test_days-i)+1:
                        len(data)-y_seq_measurements-(test_days-i)+1]
        y_slice = data[len(data)-y_seq_measurements-(test_days-i)+1:len(data)-(test_days-i)+1]

        x_test = torch.vstack((x_test, x_slice))
        y_test = torch.vstack((y_test, y_slice))

    # Original version of y_test
    y_test_raw = y_test

    # Normalize in range [-1, 1]
    x_train_min = torch.min(x_train.view(-1))
    x_train_max = torch.max(x_train.view(-1))

    y_train_min = torch.min(y_train.view(-1))
    y_train_max = torch.max(y_train.view(-1))

    x_train = 2*(x_train-x_train_min)/(x_train_max-x_train_min) - 1
    y_train = 2*(y_train-y_train_min)/(y_train_max-y_train_min) - 1

    x_test = 2*(x_test-x_train_min)/(x_train_max-x_train_min) - 1
    y_test = 2*(y_test-y_train_min)/(y_train_max-y_train_min) - 1

    # Will be useful later for denormalization
    # This is used only when evaluating the optimal model
    if results_dir_name is not None:
        save_normalization_settings(y_train_min, y_train_max, results_dir_name)

    return x_train, y_train, x_test, y_test, y_test_raw


def denormalized_preds(y_pred, target_dir_name):
    """Transform standardized data back to original scale.

    The process includes doing the inverse transformations of the ones used during data
    preprocessing. That is, the data is shifted back to its original place and de-standardized.

    Args:
        y_pred: A torch tensor that contains the model predictions.
        target_dir_name: A string with the name of the directory the transformation settings
            are saved.
    Returns:
        A torch tensors that contains the predictions in the original scale.
    """

    settings_filepath = target_dir_name + 'settings.csv'
    settings = pd.read_csv(settings_filepath)

    y_min, y_max = settings['y_min'][0], settings['y_max'][0]

    # De-normalize back to original scale
    y_pred_raw = (y_pred+1)*(y_max-y_min)/2 + y_min

    return y_pred_raw
