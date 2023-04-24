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

# TODO: Create set_cuda_reproducibility function for the baselines too
