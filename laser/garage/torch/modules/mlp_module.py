"""MLP Module."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from laser.garage.torch import NonLinearity


class MultiHeadedMLPModule(nn.Module):
    """MultiHeadedMLPModule Model.

    A PyTorch module composed only of a multi-layer perceptron (MLP) with
    multiple parallel output layers which maps real-valued inputs to
    real-valued outputs. The length of outputs is n_heads and shape of each
    output element is depend on each output dimension

    Args:
        n_heads (int): Number of different output layers
        input_dim (int): Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module or list or tuple):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearities (callable or torch.nn.Module or list or tuple):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        output_w_inits (callable or list or tuple): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        output_b_inits (callable or list or tuple): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 n_heads,
                 input_dim,
                 output_dims,
                 hidden_sizes,
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearities=None,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()

        output_dims = self._check_parameter_for_output_layer(
            'output_dims', output_dims, n_heads)
        output_w_inits = self._check_parameter_for_output_layer(
            'output_w_inits', output_w_inits, n_heads)
        output_b_inits = self._check_parameter_for_output_layer(
            'output_b_inits', output_b_inits, n_heads)
        output_nonlinearities = self._check_parameter_for_output_layer(
            'output_nonlinearities', output_nonlinearities, n_heads)

        self._layers = nn.ModuleList()
        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            if layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(prev_size))
            linear_layer = nn.Linear(prev_size, size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))

            self._layers.append(hidden_layers)
            prev_size = size

        self._output_layers = nn.ModuleList()
        for i in range(n_heads):
            output_layer = nn.Sequential()
            linear_layer = nn.Linear(prev_size, output_dims[i])
            output_w_inits[i](linear_layer.weight)
            output_b_inits[i](linear_layer.bias)
            output_layer.add_module('linear', linear_layer)

            if output_nonlinearities[i]:
                output_layer.add_module('non_linearity',
                                        NonLinearity(output_nonlinearities[i]))

            self._output_layers.append(output_layer)

    @classmethod
    def _check_parameter_for_output_layer(cls, var_name, var, n_heads):
        """Check input parameters for output layer are valid.

        Args:
            var_name (str): variable name
            var (any): variable to be checked
            n_heads (int): number of head

        Returns:
            list: list of variables (length of n_heads)

        Raises:
            ValueError: if the variable is a list but length of the variable
                is not equal to n_heads

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * n_heads
            if len(var) == n_heads:
                return var
            msg = ('{} should be either an integer or a collection of length '
                   'n_heads ({}), but {} provided.')
            raise ValueError(msg.format(var_name, n_heads, var))
        return [copy.deepcopy(var) for _ in range(n_heads)]

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return [output_layer(x) for output_layer in self._output_layers]


class MLPModule(MultiHeadedMLPModule):
    """MLP Model.

    A Pytorch module composed only of a multi-layer perceptron (MLP), which
    maps real-valued inputs to real-valued outputs.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation function
            for intermediate dense layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable or torch.nn.Module): Activation function
            for output dense layer. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__(1, input_dim, output_dim, hidden_sizes,
                         hidden_nonlinearity, hidden_w_init, hidden_b_init,
                         output_nonlinearity, output_w_init, output_b_init,
                         layer_normalization)

        self._output_dim = output_dim

    # pylint: disable=arguments-differ
    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            torch.Tensor: Output value

        """
        return super().forward(input_value)[0]

    @property
    def output_dim(self):
        """Return output dimension of network.

        Returns:
            int: Output dimension of network.

        """
        return self._output_dim


class MultiInputMLPModule(nn.Module):
    def __init__(self,
                 n_inputs,
                 input_dims,
                 output_dim,
                 hidden_sizes,
                 input_nonlinearity=torch.relu,
                 input_w_inits=nn.init.xavier_normal_,
                 input_b_inits=nn.init.zeros_,
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 layer_normalization=False):
        super().__init__()

        assert len(hidden_sizes) > 0

        input_dims = self._check_parameter_for_input_layer(
            'output_dims', input_dims, n_inputs)
        input_w_inits = self._check_parameter_for_input_layer(
            'output_w_inits', input_w_inits, n_inputs)
        input_b_inits = self._check_parameter_for_input_layer(
            'output_b_inits', input_b_inits, n_inputs)
        input_nonlinearity = self._check_parameter_for_input_layer(
            'input_nonlinearities', input_nonlinearity, n_inputs)

        input_nonlinearity = [_get_activation_fn(activation) for activation in input_nonlinearity]
        hidden_nonlinearity = _get_activation_fn(hidden_nonlinearity)
        output_nonlinearity = _get_activation_fn(output_nonlinearity)

        self._input_layers = nn.ModuleList()
        next_size = hidden_sizes[0]
        for i in range(n_inputs):
            input_layer = nn.Sequential()
            linear_layer = nn.Linear(input_dims[i], next_size)
            input_w_inits[i](linear_layer.weight)
            input_b_inits[i](linear_layer.bias)
            input_layer.add_module('linear', linear_layer)

            if input_nonlinearity[i]:
                input_layer.add_module('non_linearity',
                                       NonLinearity(input_nonlinearity[i]))

            self._input_layers.append(input_layer)

        self._shared_layers = nn.ModuleList()
        sizes = hidden_sizes[1:] + [output_dim]
        prev_size = n_inputs * next_size
        for i, size in enumerate(sizes):
            hidden_layers = nn.Sequential()
            if i < len(sizes) - 1 and layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(prev_size))
            linear_layer = nn.Linear(prev_size, size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)

            if i < len(sizes) - 1:
                if hidden_nonlinearity:
                    hidden_layers.add_module('non_linearity',
                                             NonLinearity(hidden_nonlinearity))
            else:
                if output_nonlinearity:
                    hidden_layers.add_module('non_linearity',
                                             NonLinearity(output_nonlinearity))

            self._shared_layers.append(hidden_layers)
            prev_size = size

    def forward(self, inputs):
        x = []
        for i in range(len(self._input_layers)):
            x.append(self._input_layers[i](inputs[i]))

        x = torch.cat(x, dim=-1)
        for layer in self._shared_layers:
            x = layer(x)

        return x

    @classmethod
    def _check_parameter_for_input_layer(cls, var_name, var, n_inputs):
        """Check input parameters for input layer are valid.

        Args:
            var_name (str): variable name
            var (any): variable to be checked
            n_inputs (int): number of head

        Returns:
            list: list of variables (length of n_inputs)

        Raises:
            ValueError: if the variable is a list but length of the variable
                is not equal to n_inputs

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * n_inputs
            if len(var) == n_inputs:
                return var
            msg = ('{} should be either an integer or a collection of length '
                   'n_inputs ({}), but {} provided.')
            raise ValueError(msg.format(var_name, n_inputs, var))
        return [copy.deepcopy(var) for _ in range(n_inputs)]


class MultiInputMultiHeadedMLPModule(nn.Module):
    def __init__(self, multi_input_module, multi_headed_module):
        super().__init__()
        self._multi_input_module = multi_input_module
        self._multi_headed_module = multi_headed_module

    def forward(self, inputs):
        x = self._multi_input_module(inputs)
        return self._multi_headed_module(x)


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    elif activation == 'tanh':
        return F.tanh
    elif activation is None:
        return None

    raise RuntimeError("activation should be relu/gelu/tanh or None, not {}".format(activation))