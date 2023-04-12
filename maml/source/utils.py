"""Collection of utility functions.

These are functions that surround the training/fine-tuning/hyperparameter-tuning processes and
facilitate them but do not contain any if the main logic of them.
"""

import os
import random

import numpy as np
from matplotlib import pyplot as plt
import torch


def set_cuda_reproducibility():
    """Make LSTMs deterministic when executed on GPUs to ensure reproducibility.

    See warning at https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """

    if torch.cuda.is_available():
        if torch.version.cuda == "10.1":
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        elif torch.version.cuda >= "10.2":
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"


def set_random_seeds(seed):
    """Set random seeds in all libraries that might be invoked to produce random numbers."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_device():
    """Set torch device to the best available option (cuda gpu > cpu).

    Returns:
        A string that defines the device code should be executed at
        or data should be transferred to.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def set_model_args(config):
    """Create a dictionary that contains the parameters to be fed into the Meta-Learning model.

    Args:
        config: A dictionary that contains the configuration of the whole pipeline, part of
            are the parameters of the Meta-Learning model.
    """

    args = {
        'train_epochs': config['train_epochs'],
        'task_batch_size': config['task_batch_size'],
        'sample_batch_size': config['sample_batch_size'],
        'lstm_hidden_units': config['lstm_hidden_units'],
        'init_learning_rate': float(config['init_learning_rate']),
        'meta_learning_rate': float(config['meta_learning_rate']),
        'eta_min': float(config['eta_min']),
        'num_inner_steps': config['num_inner_steps'],
        'second_order': config['second_order'],
        'second_to_first_order_epoch': config['second_to_first_order_epoch']
    }

    return args


def plot_predictions(y_true, y_pred, results_dir_name, timeseries_code):
    """Plot predicted vs true values of the given test set.

    Both true and predicted values are normalized and the plot is saved as a png file.

    Args:
        y_true: A list that contains the true output values of the examined test set.
        y_pred: A list that contains the predicted output values of the examined test set.
        results_dir_name: A string with the name of the directory the results will be saved.
        timeseries_code: A list with a string that is the id of the examined timeseries.
    """

    # Transform list of lists to list
    y_pred = [item for sublist in y_pred for item in sublist]

    plt.figure()
    plt.plot(y_true, 'b', label='True')
    plt.plot(y_pred, 'r', label='Predicted')
    plt.xlabel('Timestep')
    plt.ylabel('Normalized Load')
    plt.title('Model predictions vs true values on test set')
    plt.legend()

    target_file = results_dir_name + timeseries_code[0] + '/predictions.png'
    plt.savefig(target_file)


def get_task_test_set(test_dataloader):
    """Recreate a test set given its dataloader.

    All samples yielded by the dataloader are concatenated together and transformed to a
    single list.

    Args:
        test_dataloader: A torch Dataloader object that corresponds to a task's test set.
    Returns:
        A list that contains the test set of a specific task.
    """

    # Concatenate all output samples together.
    y_test = []
    for _, y_sample in test_dataloader:
        y_test.append(y_sample)

    # Transform list of lists to list (needs to be done twice).
    y_test = [item.tolist() for item in y_test]
    for _ in range(2):
        y_test = [item for sublist in y_test for item in sublist]

    return y_test


def plot_learning_curve(train_losses, test_losses, results_dir_name, timeseries_code):
    """Plot the learning curve of the desired model for a specific test task.

    The plot contains the train loss and the test loss of the model for the number
    of inner loop steps. The plot is saved as a png file.

    Args:
        train_losses: A list that contains the support set loss of each inner loop step for a
            specific task.
        test_losses: A list that contains the query set loss of each inner loop step for a
            specific task.
        results_dir_name: A string with the name of the directory the results will be saved.
        timeseries_code: A list with a string that is the id of the examined timeseries.
    """

    plt.figure()
    plt.plot(train_losses, c='b', label='Support Set Loss')
    plt.plot(test_losses, c='r', label='Query Set Loss')
    plt.xlabel('Inner Loop Step')
    plt.ylabel('MSE')
    plt.title('Learning curve of optimal model')
    plt.legend()

    target_dir = results_dir_name + timeseries_code[0] + "/"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    target_file = target_dir + 'learning_curve.png'
    plt.savefig(target_file)


def plot_meta_train_losses(support_losses, query_losses, results_dir_name):
    """Plot the learning curve of the model during meta-training.

    The plot contains the mean loss of all tasks' support sets the model sees in each epoch
    during meta-training. Similarly for all tasks' query sets. The plot is saved as a png file.

    Args:
        support_losses: A list that contains the mean loss for all meta-train tasks' support sets
            during each epoch.
        quer_losses: A list that contains the mean loss for all meta-train tasks' query sets
            during each epoch.
        results_dir_name: A string with the name of the directory the results will be saved.
    """

    target_file = results_dir_name + 'optimal_train_losses.png'

    plt.figure()
    plt.plot(support_losses, c='b', label='Mean support set loss')
    plt.plot(query_losses, c='r', label='Mean query set loss')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.title('Mean loss of optimal model on support and query sets of train tasks')
    plt.legend()
    plt.savefig(target_file)
