""" Module that contains the model used for embedding the tasks.

Specifically two classes that represent the two components of the recurrent autoencoder (encoder
and decoder) are defined as well as the class that implements the whole recurrent autoencoder.
Finally, a function that builds the model and renders it ready for training is also defined.
"""

import torch
from torch import nn


class Encoder(nn.Module):
    """Custom Encoder model that implements an LSTM-based encoder.

    The model is stateful, meaning that the cell and hidden states of the previous batch are
    used as inputs for the next batch. This way, the model is able to better learn long-term
    dependencies in the dataset.

    Attributes:
        input_shape: An integer that defines the number of features in the input vector.
        hidden_units: An integer that defines the number of units in the LSTM layer.
        lstm: A default torch nn.LSTM layer.
        h_n: A torch.Tensor that represents the LSTM's hidden state.
        c_n: A torch.Tensor that represents the LSTM's cell state.
        device: A string that defines the device on which the model should be loaded onto.
    """

    def __init__(self, input_shape, hidden_units, device):
        """Init Encoder with specified input shape and number of LSTM units."""

        super().__init__()

        self.hidden_units = hidden_units
        self.input_shape = input_shape

        self.lstm = nn.LSTM(input_shape, self.hidden_units, batch_first=True)

        self.device = device

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)


    def reset_states(self):
        """Reset the hidden and cell state of the model."""

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)


    def forward(self, x_sample):
        """Forward pass of the model on the given sample."""

        # Add an extra dimension so that the input is in a batch format.
        network_input = x_sample.view(x_sample.shape[0], x_sample.shape[1], self.input_shape)

        # The hidden and cell states should be detached since we do not want to learn their
        # values through back-propagation, but rather keep the values that occured from the
        # previously seen samples (due to the samples being fed sequentially).
        self.h_n, self.c_n = self.h_n.detach(), self.c_n.detach()

        output, (self.h_n, self.c_n) = self.lstm(network_input, (self.h_n, self.c_n))
        output = output.view(-1, self.hidden_units)

        # TODO: Test which one is better
        output = output[-1].unsqueeze(dim=0)
        # output = torch.unsqueeze(torch.mean(output, axis=0), dim=0)

        return output , self.h_n, self.c_n


class Decoder(nn.Module):
    """Custom Decoder model that implements an LSTM-based decoder.

    The model is stateful, meaning that the cell and hidden states of the previous batch are
    used as inputs for the next batch. This way, the model is able to better learn long-term
    dependencies in the dataset. Apart form the LSTM layer, the model also contains a linear
    layer right before the output.

    Attributes:
        input_shape: An integer that defines the number of features in the input vector.
        output_shape: An integer that defines the number of measurements in the predicted
            subsequence.
        hidden_units: An integer that defines the number of units in the LSTM layer.
        lstm: A default torch nn.LSTM layer.
        linear: A default torch nn.Linear layer.
        h_n: A torch.Tensor that represents the LSTM's hidden state.
        c_n: A torch.Tensor that represents the LSTM's cell state.
        device: A string that defines the device on which the model should be loaded onto.
    """

    def __init__(self, input_shape, output_shape, hidden_units, device):
        """Init Encoder with specified input/output shape and number of LSTM units."""

        super().__init__()

        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.lstm = nn.LSTM(input_shape, self.hidden_units, batch_first=True)
        self.linear = nn.Linear(self.hidden_units, self.output_shape)

        self.device = device

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)


    def set_states(self, h_n, c_n):
        """Set the cell and hidden states of the model with the provided values.

        Args:
            h_n: A tensor that contains the new values for the model's hidden state.
            c_n: A tensor that contains the new values for the model's cell state.
        """

        self.h_n = h_n
        self.c_n = c_n


    def forward(self, x_sample):
        """Forward pass of the model on the given sample."""

        # Add an extra dimension so that the input is in a batch format.
        network_input = x_sample.view(x_sample.shape[0], x_sample.shape[1], self.input_shape)

        # The hidden and cell states should be detached since we do not want to learn their
        # values through back-propagation, but rather keep the values that occured from the
        # previously seen samples (due to the samples being fed sequentially).
        self.h_n, self.c_n = self.h_n.detach(), self.c_n.detach()

        output, (self.h_n, self.c_n) = self.lstm(network_input, (self.h_n, self.c_n))
        output = output.view(-1, self.hidden_units)
        output = self.linear(output)[-1].unsqueeze(dim=0)
        return output


class RAE(nn.Module):  # pylint: disable=abstract-method
    """Custom Recurrent Auotencoder model that constitutes of an encoder and a decoder.

    Both the encoder and the decoder are LSTM-based. A unified forward method that contains
    the forward pass for both the encoder and the decoder together is not implemented since
    there are two types of forward passes for the model. One is the full forward pass during
    training where the aim is to reconstruct the input, the other is the encoder only forward
    pass during inference, where the aim is to obtain the embeddings of the tasks.

    Attributes:
        input_shape: An integer that defines the number of features in the input vector.
        output_shape: An integer that defines the number of measurements in the predicted
            subsequence.
        hidden_units: An integer that defines the number of units in the LSTM-based encoder
            and decoder.
        device: A string that defines the device on which the model should be loaded onto.
        encoder: A custom Encoder object that defines the encoder of the autoencoder.
        decoder: A custom Decoder object that defines the decoder of the autoencoder.
    """

    def __init__(self, input_shape, output_shape, hidden_units, device):
        """Init RAE with specified input/output shape and number of LSTM units."""

        super().__init__()

        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.device = device

        self.encoder = Encoder(self.input_shape, self.hidden_units, self.device)
        self.decoder = Decoder(self.input_shape, self.output_shape, self.hidden_units, self.device)


    def encoder_forward(self, x_sample):
        """Forward pass of the encoder on the given sample."""

        return self.encoder(x_sample)


    def decoder_forward(self, x_sample):
        """Forward pass of the decoder on the given sample."""

        return self.decoder(x_sample)


    def encoder_reset_states(self):
        """Reset the hidden and cell state of the encoder."""

        self.encoder.reset_states()


    def decoder_set_states(self, h_n, c_n):
        """Set the cell and hidden states of the decoder with the provided values.

        Args:
            h_n: A tensor that contains the new values for the decoder's hidden state.
            c_n: A tensor that contains the new values for the decoder's cell state.
        """

        self.decoder.set_states(h_n, c_n)


def build_network(input_shape, output_shape, hidden_units, device):
    """Initialize the custom RAE model and load it to the spicified device for training.

    Args:
        input_shape: An integer that defines the number of features in the input vector.
        output_shape: An integer that defines the number of measurements in the predicted
            subsequence.
        hidden_units: An integer that defines the number of units in the LSTM layer.
        device: A string that defines the device on which the model should loaded onto.
    Returns:
        An initialized custom RAE model ready to be trained.
    """

    network = RAE(input_shape,output_shape, hidden_units, device).to(device)
    return network
