import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _VF


def extract_linear_dict(current_dict, layer_id):
    linear_dict = {}
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level != 'lstm':
            if int(top_level[-1]) == layer_id:
                linear_dict[sub_level] = current_dict[key]

    return linear_dict


def extract_lstm_dict(current_dict):
    output_dict = {}
    for key in current_dict.keys():
        new_key = key.split(".")[-1]

        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]

        if top_level == 'lstm':
            output_dict[new_key] = current_dict[key]

    return output_dict


class MetaLinearLayer(nn.Module):

    def __init__(self, input_shape, output_shape, id):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.id = id

        self.weights = nn.Parameter(torch.ones(self.output_shape, self.input_shape))
        nn.init.xavier_uniform_(self.weights)
    
        self.bias = nn.Parameter(torch.zeros(output_shape))


    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """

        if params is not None:
            params = extract_linear_dict(current_dict=params, layer_id=self.id)
            weight, bias = params["weights"], params["bias"]
            weight = torch.squeeze(weight)
            bias = torch.squeeze(bias)
        else:
            weight, bias = self.weights, self.bias

        return F.linear(input=x, weight=weight, bias=bias)


class MetaLSTMLayer(nn.Module):

    def __init__(self, hidden_units, input_shape, device):
        super(MetaLSTMLayer, self).__init__()
        self.hidden_units = hidden_units
        self.input_shape = input_shape

        self.lstm = nn.LSTM(self.input_shape, self.hidden_units, batch_first=True)
        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)

        self.device = device


    def reset_states(self):
        """Reset the hidden and cell state of the model."""

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)


    def forward(self, x_sample, state_dict=None):
        if state_dict is not None:
            new_state_dict = extract_lstm_dict(state_dict)
        else:
            new_state_dict = self.lstm.load_state_dict()

        # Add an extra dimension so that the input is in a batch format.
        network_input = x_sample.view(x_sample.shape[0], x_sample.shape[1], self.input_shape)

        # TODO: Revisit this part because we have to consider both meta-training and
        # training within a task
        if self.training is True:
            # The hidden and cell states should be detached since we do not want to learn their
            # values through back-propagation, but rather keep the values that occured from the
            # previously seen samples (due to the samples being fed sequentially).
            self.h_n, self.c_n = self.h_n.detach(), self.c_n.detach()
        else:
            # During inference, we want to reset the model's hidden and cell states, since the
            # samples to be inferred are not necessarily fed sequentially to the model.
            self.reset_states()

        output, (self.h_n, self.c_n) = nn.utils.stateless.functional_call(self.lstm, 
                                                                          new_state_dict, 
                                                                          (network_input, 
                                                                           (self.h_n, self.c_n)))
        output = output.view(-1, self.hidden_units)

        return output
    

class MetaLSTMLayer2(nn.Module):

    def __init__(self, hidden_units, input_shape, device):
        super(MetaLSTMLayer2, self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.device = device

        self.weight_ih_l0 = nn.Parameter(torch.Tensor(4*self.hidden_units, self.input_shape))
        self.weight_hh_l0 = nn.Parameter(torch.Tensor(4*self.hidden_units, self.hidden_units))
        self.bias_ih_l0 = nn.Parameter(torch.Tensor(4*self.hidden_units))
        self.bias_hh_l0 = nn.Parameter(torch.Tensor(4*self.hidden_units))

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)

        self.init_parameters()


    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_units)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


    def reset_states(self):
        """Reset the hidden and cell state of the model."""

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)


    def _get_flat_weights(self, params=None):
        if params is not None:
            new_state_dict = extract_lstm_dict(params)
            return [new_state_dict['weight_ih_l0'],
                    new_state_dict['weight_hh_l0'],
                    new_state_dict['bias_ih_l0'],
                    new_state_dict['bias_hh_l0']]
        else:
            return [self.weight_ih_l0,
                    self.weight_hh_l0,
                    self.bias_ih_l0,
                    self.bias_hh_l0]


    def forward(self, x_sample, params=None, is_query_set=False):
        if params is not None:
            weights = self._get_flat_weights(params)
        else:
            weights = self._get_flat_weights()

        # Add an extra dimension so that the input is in a batch format.
        network_input = x_sample.view(x_sample.shape[0], x_sample.shape[1], self.input_shape)

        # TODO: Revisit this part because we have to consider both meta-training and
        # training within a task
        if self.training is True:
            # The hidden and cell states should be detached since we do not want to learn their
            # values through back-propagation, but rather keep the values that occured from the
            # previously seen samples (due to the samples being fed sequentially).
            self.h_n, self.c_n = self.h_n.detach(), self.c_n.detach()
        else:
            # During inference, we want to reset the model's hidden and cell states, since the
            # samples to be inferred are not necessarily fed sequentially to the model.
            self.reset_states()

        if is_query_set:
            self.reset_states()

        hx = (self.h_n, self.c_n)

        # Params: input, (h_n, c_n), weights, bias_flag, num_layers, dropout, training_flag,
        # bidirectional_flag, batch_first_flag
        output, self.h_n, self.c_n = _VF.lstm(network_input, 
                                                hx,
                                                weights,
                                                True,
                                                1,
                                                0.0,
                                                self.training,
                                                False,
                                                True)
        
        return output.view(-1, self.hidden_units)


class BaseLearner(nn.Module):

    def __init__(self,
                 input_shape,
                 output_shape,
                 hidden_units,
                 device,
                 meta_classifier=True):
        super(BaseLearner, self).__init__()

        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.device = device
        self.meta_classifier = meta_classifier

        self.layer_dict = nn.ModuleDict()
        self.layer_dict['lstm'] = MetaLSTMLayer2(self.hidden_units,
                                                self.input_shape,
                                                self.device)
        
        self.layer_dict['linear_1'] = MetaLinearLayer(self.hidden_units,
                                                      self.output_shape,
                                                      1
                                                      )

        self.layer_dict['linear_2'] = MetaLinearLayer(self.output_shape,
                                                      self.output_shape,
                                                      2
                                                      )


    def forward(self, x_sample, params=None, is_query_set=False):

        output = self.layer_dict['lstm'](x_sample, params, is_query_set)
        output = self.layer_dict['linear_1'](output, params)
        output = self.layer_dict['linear_2'](output, params)
        return output[-1].unsqueeze(dim=0)


    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if (
                    param.requires_grad == True
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    param.grad.zero_()
        else:
            for name, param in params.items():
                if (
                    param.requires_grad == True
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    param.grad.zero_()
                    params[name].grad = None


class BaseLearner2(nn.Module):

    def __init__(self,
                 input_shape,
                 output_shape,
                 hidden_units,
                 device,
                 meta_classifier=True):

        super(BaseLearner2, self).__init__()

        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.device = device
        self.meta_classifier = meta_classifier

        self.layer_dict = nn.ModuleDict()

        self.layer_dict['lstm'] = nn.LSTM(input_shape, self.hidden_units, batch_first=True)
        self.layer_dict['linear_1'] = nn.Linear(self.hidden_units, self.output_shape)
        self.layer_dict['linear_2'] = nn.Linear(self.output_shape, self.output_shape)

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)

    
    def reset_states(self):
        """Reset the hidden and cell state of the model."""

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)


    def forward(self, x_sample):

        # Add an extra dimension so that the input is in a batch format.
        network_input = x_sample.view(x_sample.shape[0], x_sample.shape[1], self.input_shape)

        # TODO: Revisit this part because we have to consider both meta-training and
        # training within a task
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
        output = self.layer_dict['linear_1'](output)
        output = self.layer_dict['linear_2'](output)

        # return output[-1].unsqueeze(dim=0)

        return output[-1].unsqueeze_(dim=0)


    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if (
                    param.requires_grad == True
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    print(param.grad)
                    param.grad.zero_()
        else:
            for name, param in params.items():
                if (
                    param.requires_grad == True
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    print(param.grad)
                    param.grad.zero_()
                    params[name].grad = None


def build_network(input_shape, output_shape, hidden_units, device, meta_classifier):
    network = BaseLearner(input_shape, output_shape, hidden_units, device, meta_classifier).to(device)
    return network
