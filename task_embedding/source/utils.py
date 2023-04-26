import os
import random

import numpy as np
from matplotlib import pyplot as plt
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


def visualize_embeddings(train_task_embeddings,
                         test_task_embeddings,
                         embedding_size,
                         results_dir_name,
                         perplexity):

    train_embeddings = np.empty((embedding_size,))
    for embedding in train_task_embeddings.values():
        train_embeddings = np.vstack([train_embeddings, np.array(embedding)])

    test_embeddings = np.empty((embedding_size,))
    for embedding in test_task_embeddings.values():
        test_embeddings = np.vstack([test_embeddings, np.array(embedding)])

    pca = PCA(n_components=30)
    train_embeddings_pca = pca.fit_transform(train_embeddings)
    test_embeddings_pca = pca.transform(test_embeddings)

    total_embeddings_pca = np.vstack([train_embeddings_pca, test_embeddings_pca])
    total_embeddings_tsne = TSNE(n_components=2,
                                 learning_rate='auto',
                                 perplexity=perplexity).fit_transform(total_embeddings_pca)

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


# TODO: Create set_cuda_reproducibility function for the baselines too
