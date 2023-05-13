"""Collection of utility functions.

These are functions that surround the training/hyperparameter-tuning process and
facilitate it but do not contain any if the main logic of it.
"""

import os

from matplotlib import pyplot as plt
import torch


def set_device():
    """Set torch device to the best available option (cuda gpu > cpu).

    Returns:
        A string that defines the device code should be executed at or data should be
        transferred to.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def set_cuda_reproducibility():
    """Make LSTMs deterministic when executed on GPUs to ensure reproducibility.

    See warning at https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """

    if torch.cuda.is_available():
        if torch.version.cuda == "10.1":
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        elif torch.version.cuda >= "10.2":
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"


def save_optimal_model(network, results_dir_name):
    """Save the best model trained.

    Define the target file in which the model will be saved and save it as a .pth file.

    Args:
        network: A PyTorch model based on torch.nn.Module.
        results_dir_name: A string with the name of the directory the results will be saved.
    """

    target_file = results_dir_name + 'optimal_model.pth'
    torch.save(obj=network.state_dict(), f=target_file)


def save_validation_logs(val_logs, results_dir_name):
    """Save the validation logs as a .csv file.

    Args:
        val_logs: A pandas DataFrame that contains the logs for the evaluated task.
        results_dir_name: A string with the name of the directory the results will be saved.
    """

    target_file = results_dir_name + 'logs.csv'
    val_logs.to_csv(target_file, index=False)


def plot_learning_curve(train_losses, test_losses,results_dir_name):
    """Plot the learning curve of the desired model.

    The plot contains the train loss and the test loss of the model for the number
    of epochs that it has been trained for. The plot is saved as a png file.

    Args:
        train_losses: A dictionary that contains the train loss of each training epoch.
        test_losses: A dictionary that contains the test loss of each training epoch.
        results_dir_name: A string with the name of the directory the results will be saved.
    """

    plt.figure()
    plt.plot(list(train_losses.values()), c='b', label='Train Loss')
    plt.plot(list(test_losses.values()), c='r', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Learning curve of optimal model')
    plt.legend()

    target_file = results_dir_name + 'learning_curve.png'
    plt.savefig(target_file)


def plot_predictions(y_true, y_pred, results_dir_name):
    """Plot predicted vs true values of the given test set.

    Both true and predicted values are normalized and the plot is saved as a png file.

    Args:
        y_true: A torch.Tensor that contains the true output values of the examined test set.
        y_pred: A list that contains the predicted output values of the examined test set.
        results_dir_name: A string with the name of the directory the results will be saved.
    """

    # Convert list of tensors to single list
    y_true = y_true.view((-1,)).tolist()

    # Convert list of lists to single list
    y_pred = [item for sublist in y_pred for item in sublist]

    plt.figure()
    plt.plot(y_true, 'b', label='True')
    plt.plot(y_pred, 'r', label='Predicted')
    plt.xlabel('Timestep')
    plt.ylabel('Normalized Load')
    plt.title('Model predictions vs true values on test set')
    plt.legend()

    target_file = results_dir_name + 'predictions.png'
    plt.savefig(target_file)
