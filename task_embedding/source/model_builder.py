import torch
from torch import nn


class Encoder(nn.Module):


    def __init__(self, input_shape, hidden_units, device):

        super().__init__()

        self.hidden_units = hidden_units
        self.input_shape = input_shape

        self.lstm = nn.LSTM(input_shape, self.hidden_units, batch_first=True)

        self.device = device

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)


    def reset_states(self):
        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)


    def forward(self, x_sample):
        network_input = x_sample.view(x_sample.shape[0], x_sample.shape[1], self.input_shape)

        self.h_n, self.c_n = self.h_n.detach(), self.c_n.detach()

        output, (self.h_n, self.c_n) = self.lstm(network_input, (self.h_n, self.c_n))
        output = output.view(-1, self.hidden_units)

        return output , self.h_n, self.c_n


class Decoder(nn.Module):


    def __init__(self, input_shape, output_shape, hidden_units, device):

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
        self.h_n = h_n
        self.c_n = c_n


    def forward(self, x_sample):
        network_input = x_sample.view(x_sample.shape[0], x_sample.shape[1], self.input_shape)

        self.h_n, self.c_n = self.h_n.detach(), self.c_n.detach()

        output, (self.h_n, self.c_n) = self.lstm(network_input, (self.h_n, self.c_n))
        output = output.view(-1, self.hidden_units)
        output = self.linear(output)[-1].unsqueeze(dim=0)
        return output


class RAE(nn.Module):  # pylint: disable=abstract-method


    def __init__(self, input_shape, output_shape, hidden_units, device):
        super().__init__()

        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.device = device

        self.encoder = Encoder(self.input_shape, self.hidden_units, self.device)
        self.decoder = Decoder(self.input_shape, self.output_shape, self.hidden_units, self.device)


    def encoder_forward(self, x_sample):
        return self.encoder(x_sample)


    def decoder_forward(self, x_sample):
        return self.decoder(x_sample)


    def encoder_reset_states(self):
        self.encoder.reset_states()


    def decoder_set_states(self, h_n, c_n):
        self.decoder.set_states(h_n, c_n)


def build_network(input_shape, output_shape, hidden_units, device):

    network = RAE(input_shape,output_shape, hidden_units, device).to(device)
    return network
