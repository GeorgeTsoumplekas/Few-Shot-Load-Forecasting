"""Module that contains the different loss functions used during training/evaluation.

Spefically, the MAPE and MALPE (custom made loss metric) functions are defined as well as
a function that selects the loss function to be used.
"""

import torch
from torch import nn


def MAPE(y_pred, y_test):
    """Calculates the MAPE loss for the given target and prediction.

    Args:
        y_pred: A torch tensor that includes the model predictions in the original scale
        y_true: A torch tensor that includes the true ooutput values inthe original scale.

    Returns:
        A float that is the MAPE loss value.
    """

    return 100*torch.mean(torch.abs((y_pred - y_test) / y_test)).item()


def MALPE(y_pred, y_test):
    """Calculate the mean absolute percentage of the logarithmic forecast to actual values ratio.

    Args:
        y_pred: A torch tensor that includes the model predictions in the original scale
        y_true: A torch tensor that includes the true output values in the original scale.

    Returns:
        A float that is the MALPE loss value.  
    """

    return 100*torch.mean(torch.abs(torch.log(y_pred/y_test))).item()


def get_loss(loss_name):
    """Get the desired loss function.

    This is done by selecting one of the two available loss functions: MSE and MAE.

    Args:
        loss_name: A string that is the name of the oss function to be used.
    Returns:
        A class object that represents the desired loss function to be used.
    """

    if loss_name == 'MSE':
        return nn.MSELoss()
    if loss_name == 'MAE':
        return nn.L1Loss()
    raise ValueError("Not a valid loss function provided.")
