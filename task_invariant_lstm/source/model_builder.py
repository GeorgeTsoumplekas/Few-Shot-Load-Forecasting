"""Module that contains the model used for training/inference.

Specifically, a class that defines the custom LSTM model is included as well as a function that
builds the model and renders it ready for training.
"""

import torch
from torch import nn


class LSTMModel(nn.Module):
    """Custom LSTM model that constitutes of a LSTM layer followed by a linear layer.

    The model is stateful during training, meaning that the cell and hidden states of the previous
    batch are used as inputs for the next batch. This way, the model is able to better learn
    long-term dependencies in the dataset. In contrast, during inference, the model is
    stateless, meaning that the hidden and cell states are reset to zero before being used as inputs
    by the model.

    Attributes:
        input_shape: An integer that defines the number of features in the input vector.
        output_shape: An integer that defines the number of measurements in the predicted
            subsequence.
        hidden_units: An integer that defines the number of units in the LSTM layer.
        num_lin_layers: An integer that defines the number of linear layers in the model.
        layer_dict: A torch module dictionary whose entries are the layers of the model.
        h_n: A torch.Tensor that represents the LSTM's hidden state.
        c_n: A torch.Tensor that represents the LSTM's cell state.
        device: A string that defines the device on which the model should be loaded onto.
    """

    def __init__(self, input_shape, output_shape, hidden_units, num_lin_layers, device):
        """Init LSTMModel with specified input/output shape and number of LSTM units."""

        super().__init__()

        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_lin_layers = num_lin_layers

        self.layer_dict = nn.ModuleDict()
        self.layer_dict['lstm'] = nn.LSTM(input_shape, self.hidden_units, batch_first=True)

        # Create linear layers - their number varies
        for i in range(self.num_lin_layers):
            lin_layer_name = 'linear_' + str(i+1)

            # Last linear layer
            if i==self.num_lin_layers-1:
                self.layer_dict[lin_layer_name] = nn.Linear(self.hidden_units, self.output_shape)
            # Intermediate linear layers
            else:
                self.layer_dict[lin_layer_name] = nn.Linear(self.hidden_units, self.hidden_units)

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)

        self.device = device


    def reset_states(self):
        """Reset the hidden and cell state of the model."""

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)


    def forward(self, x_sample):
        """Forward pass of the model on the given sample."""

        # Add an extra dimension so that the input is in a batch format.
        network_input = x_sample.view(x_sample.shape[0], x_sample.shape[1], self.input_shape)

        if self.training is True:
            # The hidden and cell states should be detached since we do not want to learn their
            # values through back-propagation, but rather keep the values that occured from the
            # previously seen samples (due to the samples being fed sequentially).
            self.h_n, self.c_n = self.h_n.detach(), self.c_n.detach()
        else:
            # During inference, we want to reset the model's hidden and cell states, since the
            # samples to be inferred are not necessarily fed sequentially to the model.
            self.reset_states()

        output, (self.h_n, self.c_n) = self.layer_dict['lstm'](network_input, (self.h_n, self.c_n))
        output = output.view(-1, self.hidden_units)

        for i in range(self.num_lin_layers):
            lin_layer_name = 'linear_' + str(i+1)

            # Last linear layer
            if i==self.num_lin_layers-1:
                output = self.layer_dict[lin_layer_name](output)[-1].unsqueeze(dim=0)
            # Intermediate linear layers
            else:
                output = self.layer_dict[lin_layer_name](output)

        return output


def build_network(input_shape, output_shape, hidden_units, num_lin_layers, device):
    """Initialize the custom LSTM model and load it to the spicified device for training.

    Args:
        input_shape: An integer that defines the number of features in the input vector.
        output_shape: An integer that defines the number of measurements in the predicted
            subsequence.
        hidden_units: An integer that defines the number of units in the LSTM layer.
        num_lin_layers: An integer that defines the number of linear layers of the model.
        device: A string that defines the device on which the model should loaded onto.
    Returns:
        An initialized custom LSTM model ready to be trained.
    """

    network = LSTMModel(input_shape, output_shape, hidden_units, num_lin_layers, device).to(device)
    return network
