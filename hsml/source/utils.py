"""Collection of utility functions.

These are functions that surround the training/fine-tuning/hyperparameter-tuning processes and
facilitate them but do not contain any if the main logic of them.
"""

import os
import random

import numpy as np
from matplotlib import pyplot as plt
import torch


def set_device():
    """Set torch device to the best available option (cuda gpu > cpu).

    Returns:
        A string that defines the device code should be executed at
        or data should be transferred to.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device
