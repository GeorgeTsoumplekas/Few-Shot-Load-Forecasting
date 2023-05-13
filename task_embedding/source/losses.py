"""Module that contains the different loss functions used during training/evaluation.

Spefically, a class that models a loss function specific for gamma-distributed data is defined
as well as a function that selects the loss function to be used.
"""

import torch
from torch import nn


class GammaLoss(nn.Module):
    """Custom loss function appropriate for gamma-distributed data.

    The implementation is based on this thread:
    https://stats.stackexchange.com/questions/484555/loss-function-in-for-gamma-objective-function-in-regression-in-xgboost

    Attributes:
        kappa: A float that is the shape parameter of the gamma distribution.
    """

    def __init__(self, kappa, device):
        """Init GammaLoss with specific shape parameter"""

        super().__init__()

        # Should be defined as a tensor and be moved to the appropriate device because
        # torch.lgamma only accepts tensors as input.
        self.kappa = torch.tensor([kappa]).to(device)

    def forward(self, y_pred, y_true):
        """Loss value calculation given the predicted and the corresponding true values.

        Args:
            y_pred: A torch tensor that contains the predicted output values of the examined
                timeseries.
            y_true: A torch tensor that contains the true output values of the examined timeseries.
        Returns:
            A torch tensor that contains the loss for the given sample.
        """

        elementwise_losses = torch.lgamma(self.kappa) \
                             + self.kappa*y_pred \
                             - self.kappa*torch.log(self.kappa) \
                             - (self.kappa-1)*torch.log(y_true) \
                             + self.kappa*y_true*torch.exp(-y_pred)
        return torch.mean(elementwise_losses)


def get_loss(loss_name, kappa=None, device=None):
    """Get the desired loss function.

    This is done by selecting one of the three available loss functions: MSE, MAE and the custom
    GammaLoss.

    Args:
        loss_name: A string that is the name of the oss function to be used.
        kappa: Optional, a float that defines the shape paprameter of the gamma distribution used
            in the gamma loss function.
        device: Optional, a string that defines the device on which calculations take place.
    Returns:
        A class object that represents the desired loss function to be used.
    """

    if loss_name == 'MSE':
        return nn.MSELoss()
    if loss_name == 'MAE':
        return nn.L1Loss()
    if loss_name == 'Gamma':
        return GammaLoss(kappa, device)
    raise ValueError("Not a valid loss function provided.")
