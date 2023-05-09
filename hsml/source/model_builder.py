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


class HierarchicalClustering(nn.Module):
    
    def __init__(self, num_levels, num_centers, sigma, embedding_size):

        super().__init__()

        self.num_levels = num_levels
        self.num_centers = num_centers
        self.embedding_size = embedding_size
        self.sigma = sigma

        self.cluster_centers = nn.ParameterDict()
        self.linear_layers = nn.ModuleDict()

        for i in range(1, self.num_levels):
            layer_name = "linear_" + str(i)
            self.linear_layers[layer_name] = nn.Linear(in_features=self.embedding_size,
                                                       out_features=self.embedding_size,
                                                       bias=True)

        # TODO: determine how to initialize cluster centers
        for i in range(1, self.num_levels):
            for j in range(self.num_centers[i]):
                center_name = "level_" + str(i) + "_center_" + str(j+1)
                self.cluster_centers[center_name] = nn.Parameter(
                    data=torch.rand(self.embedding_size), requires_grad=True)
                

    def assignment_step(self, h_i, level):

        # level starts from 1, not 0, i.e. 1,2,3 for 4 levels
        p_i = torch.zeros(self.num_centers[level-1], self.num_centers[level])

        for i, h_i_cluster in enumerate(h_i):  # test we get a correct h_i_cluster each time
            assignment_scores = torch.zeros(self.num_centers[level])

            for center_idx in range(self.num_centers[level]):
                center_name = "level_" + str(level) + "_center_" + str(center_idx+1)

                assignment_scores[center_idx] = -torch.sum(torch.square(
                    h_i_cluster - self.cluster_centers[center_name]) /
                    (2.0 * self.sigma))

            p_i[i][:] = torch.softmax(assignment_scores, dim=0)  # test it runs ok for all levels

        return p_i


    def update_step(self, p_i, h_i, level):

        # level starts from 1, not 0, i.e. 1,2,3 for 4 levels
        h_i_next = torch.zeros(self.num_centers[level], self.embedding_size)

        for i in range(self.num_centers[level]):
            layer_name = "linear_" + str(level)

            for j in range(self.num_centers[level-1]):
                h_i_next[i] += (p_i[j][i] * torch.tanh(self.linear_layers[layer_name](h_i[j])))

        return h_i_next
 

    def forward(self, task_embedding):
        
        h_level = task_embedding

        for level in range(1, self.num_levels):
            p_level = self.assignment_step(h_level, level)
            h_level = self.update_step(p_level, h_level, level)

        return h_level
    

class ParameterGate(nn.Module):

    def __init__(self, input_shape, output_shape):

        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.linear = nn.Linear(input_shape, output_shape)
        self.sigmoid = nn.Sigmoid()


    def forward(self, task_representation, task_cluster_representation):

        task_total_representation = torch.cat([task_representation, task_cluster_representation],
                                              dim=0)
        return self.sigmoid(self.linear(task_total_representation))
    

def build_base_network(input_shape, output_shape, hidden_units, device):
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

    base_network = BaseLearner(
        input_shape,
        output_shape,
        hidden_units,
        device).to(device)
    return base_network


def build_cluster_network(num_levels, num_centers, sigma, embedding_size, device):
    cluster_network = HierarchicalClustering(num_levels,
                                             num_centers,
                                             sigma,
                                             embedding_size).to(device)
    return cluster_network


def build_parameter_gate(input_shape, output_shape, device):

    parameter_gate = ParameterGate(input_shape, output_shape).to(device)
    return parameter_gate
