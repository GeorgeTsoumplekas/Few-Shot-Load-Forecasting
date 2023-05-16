"""Collection of utility functions.

These are functions that surround the training/fine-tuning/hyperparameter-tuning processes and
facilitate them but do not contain any if the main logic of them.
"""

import os
import random

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
    """Save given model.

    Define the target file in which the model will be saved and save it as a .pth file.

    Args:
        network_state_dict: The state dictionary of the model.
        results_dir_name: A string with the name of the directory the results will be saved.
    """

    target_file = results_dir_name + 'optimal_trained_model.pth'
    torch.save(obj=network_state_dict, f=target_file)


def plot_learning_curve(train_losses, test_losses, results_dir_name, loss):
    """Plot the learning curve of the model during training.

    The plot contains the train loss and the test loss of the model for the number
    of epochs that it has been trained for. The plot is saved as a png file.

    Args:
        train_losses: A list that contains the train loss of each training epoch.
        test_losses: A list that contains the test loss of each training epoch.
        results_dir_name: A string with the name of the directory the results will be saved.
        loss: A string that is the name of the loss function used.
    """

    plt.figure()
    plt.plot(train_losses, c='b', label='Train Loss')
    plt.plot(test_losses, c='r', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel(loss)
    plt.title('Learning curve of optimal model')
    plt.legend()

    target_file = results_dir_name + 'learning_curve.png'
    plt.savefig(target_file)


def plot_predictions(y_true, y_pred, target_dir):
    """Plot predicted vs true values of the given timeseries.

    Both true and predicted values are normalized and the plot is saved as a png file.

    Args:
        y_true: A list that contains the true output values of the examined timeseries.
        y_pred: A list that contains the predicted output values of the examined timeseries.
        target_dir: A string with the name of the directory the results will be saved.
    """

    plt.figure()
    plt.plot(y_true, 'b', label='True')
    plt.plot(y_pred, 'r', label='Predicted')
    plt.xlabel('Timestep')
    plt.ylabel('Normalized Load')
    plt.title('Model predictions vs true values on test set')
    plt.legend()

    target_file = target_dir + 'reconstruction.png'
    plt.savefig(target_file)


def save_validation_logs(val_logs, target_dir):
    """Save the validation logs as a .csv file.

    Args:
        val_logs: A pandas DataFrame that contains the logs for each evaluated task.
        target_dir: A string with the name of the directory the results will be saved.
    """

    # Sort logs based on the time series code
    val_logs = val_logs.sort_values(by=['timeseries_code'])

    target_file = target_dir + 'logs.csv'
    val_logs.to_csv(target_file, index=False)


def plot_distributions(y_true, y_pred, target_dir):
    """ Plot timeseries' values and prediction's errors distributions.

    The task time series and the errors time series are transformed to pandas DataFrames and the
    corrsponding histograms are plotted and saved as .png files.

    Args:
        y_true: A torch tensor that contains the true output values of the examined timeseries.
        y_pred: A torch tensor that contains the predicted output values of the examined timeseries.
        target_dir: A string with the name of the directory the results will be saved.
    """

    errors = y_pred - y_true

    y_true = pd.DataFrame(y_true.cpu().numpy())
    y_true.rename(columns={0: 'Measurements'}, inplace=True)

    errors = pd.DataFrame(errors.cpu().numpy())
    errors.rename(columns={0: 'Measurements'}, inplace=True)

    plt.figure()
    y_true['Measurements'].plot.hist(bins=40)
    plt.title('Distribution of time series values')
    target_file = target_dir + 'values_distribution.png'
    plt.savefig(target_file)

    plt.figure()
    errors['Measurements'].plot.hist(bins=40)
    plt.title('Distribution of reconstruction errors')
    target_file = target_dir + 'errors_distribution.png'
    plt.savefig(target_file)


def visualize_embeddings(train_task_embeddings,
                         test_task_embeddings,
                         embedding_size,
                         results_dir_name,
                         perplexity):
    """Create 2D visualizations of the embeddings of the tasks.

    First, PCA is applied to reduce the dimensionality of the tasks and then tSNE is applied
    to further reduce the dimensionality to two dimensions. Fianlly, a scatter plot that contains
    the embeddings of both training and test tasks is created and saved.

    Args:
        train_task_embeddings: A dictionary that contains the embedding of each task in the
            train set.
        test_task_embeddings: A dictionary that contains the embedding of each task in the
            test set.
        embedding_size: An integer that is the length of the task embedding.
        results_dir_name: A string with the name of the directory the results will be saved.
        perplexity: A float related to the number of nearest neighbors that is used in other
            manifold learning algorithms.
    """

    # Create an array that contains the embeddings of the train set tasks.
    train_embeddings = np.empty((embedding_size,))
    for embedding in train_task_embeddings.values():
        train_embeddings = np.vstack([train_embeddings, np.array(embedding)])

    # Create an array that contains the embeddings of the test set tasks.
    test_embeddings = np.empty((embedding_size,))
    for embedding in test_task_embeddings.values():
        test_embeddings = np.vstack([test_embeddings, np.array(embedding)])

    # Apply PCA
    n_components = None
    if embedding_size<30:
        n_components = round(0.9*embedding_size)
    else:
        n_components = 30
    pca = PCA(n_components=n_components)

    train_embeddings_pca = pca.fit_transform(train_embeddings)
    test_embeddings_pca = pca.transform(test_embeddings)

    # Stack all embeddings together and apply tSNE to all of them altogether.
    total_embeddings_pca = np.vstack([train_embeddings_pca, test_embeddings_pca])
    total_embeddings_tsne = TSNE(n_components=2,
                                 learning_rate='auto',
                                 perplexity=perplexity).fit_transform(total_embeddings_pca)

    # Separate train and test tasks to their dimensions, then plot.
    x_train_embeddings_tsne = total_embeddings_tsne[:train_embeddings_pca.shape[0], 0]
    y_train_embeddings_tsne = total_embeddings_tsne[:train_embeddings_pca.shape[0], 1]

    x_test_embeddings_tsne = total_embeddings_tsne[train_embeddings_pca.shape[0]:, 0]
    y_test_embeddings_tsne = total_embeddings_tsne[train_embeddings_pca.shape[0]:, 1]

    plt.figure(figsize=(10, 8))
    plt.scatter(x_train_embeddings_tsne,
                y_train_embeddings_tsne,
                c='b',
                alpha=0.7,
                label='Train tasks')
    plt.scatter(x_test_embeddings_tsne,
                y_test_embeddings_tsne,
                c='r',
                alpha=0.7,
                label='Test tasks')
    plt.title("2D visualization of task embeddings using tSNE")
    plt.legend()

    target_file = results_dir_name + 'tsne.png'
    plt.savefig(target_file)
