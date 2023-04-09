import os
import random

import numpy as np
from matplotlib import pyplot as plt
import torch


def set_random_seeds(seed):
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
    of epochs that it has been fine-tuned for. The plot is saved as a png file.

    Args:
        train_losses: A list that contains the train loss of each fine-tune epoch.
        test_losses: A list that contains the test loss of each training epoch.
        results_dir_name: A string with the name of the directory the results will be saved.
        timeseries_code: A list with a string that is the id of the examined timeseries.
    """

    plt.figure()
    plt.plot(train_losses, c='b', label='Train Loss')
    plt.plot(test_losses, c='r', label='Test Loss')
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
    target_file = results_dir_name + 'optimal_train_losses.png'

    plt.figure()
    plt.plot(support_losses, c='b', label='Mean support set loss')
    plt.plot(query_losses, c='r', label='Mean query set loss')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.title('Loss of optimal model on support and query sets of train tasks')
    plt.legend()
    plt.savefig(target_file)
