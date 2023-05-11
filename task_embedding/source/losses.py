import torch
from torch import nn


class Gamma_Loss():

    def __init__(self, kappa):

        super().__init__()
        self.kappa = kappa

    def forward(self, y_pred, y_true):

        elementwise_losses = torch.lgamma(self.kappa) \
                             + self.kappa*torch.log(y_pred) \
                             - self.kappa*torch.log(self.kappa) \
                             - (self.kappa-1)*torch.log(y_true) \
                             + (self.kappa*y_true)/y_pred
        return torch.mean(elementwise_losses)


def get_loss(loss_name, kappa=None):

    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'MAE':
        return nn.L1Loss()
    elif loss_name == 'Gamma':
        return Gamma_Loss(kappa)
    else:
        raise ValueError("Not a valid loss function provided.")
