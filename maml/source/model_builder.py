"""Module that contains the necessary parts of the base learner.

More specifically, apart form the class that defines the base learner used inside the meta-learner,
two additional custom layers are defined as classes here: the meta-linear layer and the meta-LSTM
layer. The reason behond creating these classes is that custom pytorch layers do not allow
functional calls where the weights are user-defined. This is crucial in meta-learning where we
want to allow having two different sets of parameters: the initial ones being optimized in the outer
loop and the fine-tuned ones that are updated in each inner loop step. Additionally, two functions
to extract the state dict for each meta-layer from the total state dict of the meta-learner, as
well as a function to initialize the base learner are also defined.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch import _VF


def extract_linear_dict(current_dict, layer_id):
    """Extract the dictionary that contains the linear layer's parameters from the state dictionary
        of the whole base model.

    Args:
        current_dict: A dictionary that contains the parameters of the whole model.
        layer_id: An integer that defines whether this layer is the 1st, 2nd, etc. in the stack
            of linear layers used by the base model.
    Returns:
        A dictionary that contains the weights and biases of the specific linear layer.
    """

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
    """Extract the dictionary that contains the LSTM layer's parameters from the state dictionary
        of the whole base model.

    Args:
        current_dict: A dictionary that contains the parameters of the whole model.
    Returns:
        A dictionary that contains the weights and biases of the LSTM layer.
    """

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
    """A custom linear layer used by the base model.

    Applies the same functionality of a standard pytorch linear layer with the added functionality
    of being able to receive a parameter dictionary at the forward pass which allows the layer
    to use external weights instead of the internal ones stored inside it. Useful during inner
    loop optimization in the meta learning setting.

    Attributes:
        input_shape: An integer that represents the dimensionality of an input sample.
        output_shape: An integer that reprsents the dimensionality of the output.
        layer_id: An integer that defines whether this layer is the 1st, 2nd, etc. in the stack
            of linear layers used by the base model.
        weights: A torch Parameter tensor that contains the internal weights of the layer.
        bias: A torch Parameter tensor that contains the internal biases of the layer.
    """

    def __init__(self, input_shape, output_shape, layer_id):
        """Init MetaLinearLayer with specified input/output shape and number in the layer stack."""

        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer_id = layer_id

        # Use xavier uniform initialization
        self.weights = nn.Parameter(torch.ones(self.output_shape, self.input_shape))
        nn.init.xavier_uniform_(self.weights)

        # Biases initialized as zeros
        self.bias = nn.Parameter(torch.zeros(output_shape))


    def forward(self, x_sample, params=None):
        """Forward pass of the layer on the given sample.

        Forward propagates by applying a linear function (Wx + b). If params are none then
        internal params are used. Otherwise passed params will be used to execute the function.

        Args:
            x_sample: A Tensor that corresponds to the input sample.
            params: A dictionary that contains the state dict of the base model.
        Returns:
            A Tesnor which is the result of the linear function.
        """

        if params is not None:
            params = extract_linear_dict(current_dict=params, layer_id=self.layer_id)
            weight, bias = torch.squeeze(params["weights"]), torch.squeeze(params["bias"])

        else:
            weight, bias = self.weights, self.bias

        # Functional call with defined weights and biases.
        return F.linear(input=x_sample, weight=weight, bias=bias)


class MetaLSTMLayer(nn.Module):
    """A custom LSTM layer used by the base model.

    Applies the same functionality of a standard pytorch LSTM layer with the added functionality
    of being able to receive a parameter dictionary at the forward pass which allows the layer
    to use external weights instead of the internal ones stored inside it. Useful during inner
    loop optimization in the meta learning setting.

    The model is stateful during training, meaning that the cell and hidden states of the previous
    batch are used as inputs for the next batch. This way, the model is able to better learn
    long-term dependencies in the dataset. In contrast, during inference, the model is
    stateless, meaning that the hidden and cell states are reset to zero before being used as inputs
    by the model.

    Attributes:
        input_shape: An integer that defines the number of features in the input vector.
        hidden_units: An integer that defines the number of units in the LSTM layer.
        device: A string that defines the device on which the model should loaded onto.
        weight_ih_l0: A torch Tensor with the learnable input-hidden weights of the layer.
        weight_hh_l0: A torch Tensor with the learnable hidden-hidden weights of the layer.
        bias_ih_l0: A torch Tensor with the learnable input-hidden bias of the layer.
        bias_hh_l0: A torch Tensor with the learnable hidden-hidden bias of the layer.
        h_n: A torch Tensor that represents the LSTM's hidden state.
        c_n: A torch.Tensor that represents the LSTM's cell state.
    """

    def __init__(self, hidden_units, input_shape, device):
        """Init MetaLSTMLayer with specified input shape and number of LSTM units."""

        super().__init__()

        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.device = device

        # Define internal parameters
        self.weight_ih_l0 = nn.Parameter(torch.Tensor(4*self.hidden_units, self.input_shape))
        self.weight_hh_l0 = nn.Parameter(torch.Tensor(4*self.hidden_units, self.hidden_units))
        self.bias_ih_l0 = nn.Parameter(torch.Tensor(4*self.hidden_units))
        self.bias_hh_l0 = nn.Parameter(torch.Tensor(4*self.hidden_units))

        # Define internal states
        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(device)

        self.init_parameters()


    def init_parameters(self):
        """Initialize layer's internal parameters.

        This is done in the same way done by the in-built LSTM layer.
        See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        """

        stdv = 1.0 / math.sqrt(self.hidden_units)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


    def reset_states(self):
        """Reset the hidden and cell state of the model."""

        self.h_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)
        self.c_n = torch.zeros(1,1,self.hidden_units, dtype=torch.float32).to(self.device)


    def _get_flat_weights(self, params=None):
        """Create a list that contains all layer's weights.

        If a parameter list is given, the final list is created based on this, otherwise the
        internal weights of the layer are used.

        Args:
            params: A dictionary that contains the state dict of the base model.
        Returns:
            A list that contains all weights of the layer together.
        """

        if params is not None:
            new_state_dict = extract_lstm_dict(params)
            return [new_state_dict['weight_ih_l0'],
                    new_state_dict['weight_hh_l0'],
                    new_state_dict['bias_ih_l0'],
                    new_state_dict['bias_hh_l0']]

        return [self.weight_ih_l0,
                self.weight_hh_l0,
                self.bias_ih_l0,
                self.bias_hh_l0]


    def forward(self, x_sample, params=None, is_query_set=False):
        """Forward pass of the layer on the given sample.

        Args:
            x_sample: A Tensor that corresponds to the input sample.
            params: A dictionary that contains the state dict of the base model.
            is_query_set: A boolean that defines whether the given input sample belongs to the
                query set of the task.
        Returns:
            A Tensor which is the result of the LSTM function.
        """

        # This weight format is necassary for the _VF.lstm function
        if params is not None:
            weights = self._get_flat_weights(params)
        else:
            weights = self._get_flat_weights()

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

        # Although being in training mode, the query sets have to be treated as being in
        # evaluation mode.
        if self.training and is_query_set:
            self.reset_states()

        h_x = (self.h_n, self.c_n)

        # We use this function because pytorch does not have a functional LSTM layer.
        # Params: input, (h_n, c_n), weights, bias_flag, num_layers, dropout, training_flag,
        # bidirectional_flag, batch_first_flag
        output, self.h_n, self.c_n = _VF.lstm(network_input,  # pylint: disable=no-member
                                              h_x,
                                              weights,
                                              True,
                                              1,
                                              0.0,
                                              self.training,
                                              False,
                                              True)

        return output.view(-1, self.hidden_units)


class BaseLearner(nn.Module):
    """Custom LSTM model that constitutes of a LSTM layer followed by two linear layers.

    The model is created using the custom functional layers. As a result the model can be used in
    functional calls with different user-defined weights in each call.

    Attributes:
        input_shape: An integer that defines the number of features in the input vector.
        output_shape: An integer that defines the number of measurements in the predicted
            subsequence.
        hidden_units: An integer that defines the number of units in the LSTM layer.
        device: A string that defines the device on which the model should be loaded onto.
        layer_dict: A torch module dictionary whose entries are the layers of the base network.
    """

    def __init__(self, input_shape, output_shape, hidden_units, device):
        """Init BaseLearner with specified input/output shape and number of LSTM units."""

        super().__init__()

        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.device = device

        # Define the layers of the base learner
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['lstm'] = MetaLSTMLayer(self.hidden_units,
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


    def zero_grads(self, params=None):
        """Set the parameters of the base model to zero.

        If the params dictionary is given, then the gradients of these parameters are zero'ed.
        Otherwise, the base model's internal parameters are used.

        Args:
            params: A dictionary that contains the state dict of the base model.
        """

        if params is None:
            for param in self.parameters():
                if (
                    param.requires_grad is True
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    param.grad.zero_()
        else:
            for name, param in params.items():
                if (
                    param.requires_grad is True
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    param.grad.zero_()
                    params[name].grad = None


    def reset_states(self):
        """Reset the hidden and cell state of the model."""

        self.layer_dict['lstm'].reset_states()


    def forward(self, x_sample, params=None, is_query_set=False):
        """Forward pass of the model on the given sample.

        Args:
            x_sample: A Tensor that corresponds to the input sample.
            params: A dictionary that contains the state dict of the base model.
            is_query_set: A boolean that defines whether the given input sample belongs to the
                query set of the task.
        Returns:
            A Tensor which is the output of the base model.
        """

        output = self.layer_dict['lstm'](x_sample, params, is_query_set)
        output = self.layer_dict['linear_1'](output, params)
        output = self.layer_dict['linear_2'](output, params)
        return output[-1].unsqueeze(dim=0)


def build_network(input_shape, output_shape, hidden_units, device):
    """Initialize the custom base model and load it to the spicified device for training.

    Args:
        input_shape: An integer that defines the number of features in the input vector.
        output_shape: An integer that defines the number of measurements in the predicted
            subsequence.
        hidden_units: An integer that defines the number of units in the LSTM layer.
        device: A string that defines the device on which the model should loaded onto.
    Returns:
        An initialized custom base model ready to be used by the meta-learner.
    """

    network = BaseLearner(
        input_shape,
        output_shape,
        hidden_units,
        device).to(device)
    return network
