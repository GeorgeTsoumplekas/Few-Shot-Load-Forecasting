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


def save_model(network_state_dict, results_dir_name):

    target_file = results_dir_name + 'optimal_trained_model.pth'
    torch.save(obj=network_state_dict, f=target_file)


def plot_learning_curve(train_losses, test_losses, results_dir_name):

    plt.figure()
    plt.plot(train_losses, c='b', label='Train Loss')
    plt.plot(test_losses, c='r', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Learning curve of optimal model')
    plt.legend()

    target_file = results_dir_name + 'learning_curve.png'
    plt.savefig(target_file)


def plot_predictions(y_true, y_pred, results_dir_name, timeseries_code):
    
    plt.figure()
    plt.plot(y_true, 'b', label='True')
    plt.plot(y_pred, 'r', label='Predicted')
    plt.xlabel('Timestep')
    plt.ylabel('Normalized Load')
    plt.title('Model predictions vs true values on test set')
    plt.legend()

    target_file = results_dir_name + timeseries_code[0] + '_reconstruction.png'
    plt.savefig(target_file)


# TODO: Create set_cuda_reproducibility function for the baselines too
