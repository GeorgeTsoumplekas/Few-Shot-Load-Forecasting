import torch
from torch import nn


class Gamma_Loss(nn.Module):

    def __init__(self, kappa, device):

        super().__init__()
        self.kappa = torch.tensor([kappa]).to(device)

    def forward(self, y_pred, y_true):
        elementwise_losses = torch.lgamma(self.kappa) \
                             + self.kappa*y_pred \
                             - self.kappa*torch.log(self.kappa) \
                             - (self.kappa-1)*torch.log(y_true) \
                             + self.kappa*y_true*torch.exp(-y_pred)
        return torch.mean(elementwise_losses)


def get_loss(loss_name, kappa=None, device=None):

    if loss_name == 'MSE':
        return nn.MSELoss()
    if loss_name == 'MAE':
        return nn.L1Loss()
    if loss_name == 'Gamma':
        return Gamma_Loss(kappa, device)
    raise ValueError("Not a valid loss function provided.")
