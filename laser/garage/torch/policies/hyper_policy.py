"""
Based on the Decision Adapter:
https://github.com/Michael-Beukman/DecisionAdapter

Beukman, M., Jarvis, D., Klein, R., James, S. and Rosman, B., 2024.
Dynamics generalisation in reinforcement learning via adaptive context-aware policies.
Advances in Neural Information Processing Systems, 36.
"""

import numpy as np
import torch
import torch.nn as nn
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
from laser.garage.torch.modules.mlp_module import MLPModule

from laser.garage.torch._functions import global_device
from laser.garage.torch.policies.policy import Categorical, DiagGaussian, init, init_normc_, FixedCategorical
from laser.garage.utils import helpers as utl
from laser.garage.utils.running_stats import TorchRunningMeanStd


def forward_target_mlp(h, params, activation_fn, expected_shape, return_pre_activations=False):
    """
    Construct and forward pass an MLP, using the given parameters (weights + bias)

    :param h: Input
    :param params: Parameters (e.g. from a hypernetwork). Each batch in h will have its own set of parameters
    :return:
    """

    # Save a copy of the original input
    old_h = h

    h = h.unsqueeze(-2)
    for layer_i in range(0, len(params[0]), 2):
        # Get the weights and bias for this layer, for all batches
        w = torch.stack([params[batch][layer_i] for batch in range(len(params))])  # (p, w_in, w_out)
        b = torch.stack([params[batch][layer_i + 1] for batch in range(len(params))]).unsqueeze(1)  # (p, 1, w_out)

        if 3 < len(h.shape) != len(expected_shape):
            # Full trajectory case: the same context weights are used for each step, so duplicate it
            horizon = h.shape[-3]  # H
            w = w.unsqueeze(-3).unsqueeze(-3).repeat(1, 1, horizon, 1, 1)  # (p, q, H, w_in, w_out)
            b = b.unsqueeze(-3).unsqueeze(-3).repeat(1, 1, horizon, 1, 1)  # (p, q, H, 1, w_out)

        # activation(x*w.T + b) ==> this is similar to a torch.linear layer, but applies batch-wise weights
        h = torch.matmul(h, w) + b
        if return_pre_activations and layer_i == len(params[0]) - 2:
            pre_activations = h.clone()
        h = activation_fn(h)
    h = h.squeeze(-2)

    # Residual connection
    h = old_h + h

    if return_pre_activations:
        return h, pre_activations
    return h


class HyperPolicy(nn.Module):
    def __init__(self,
                 args,
                 # input
                 pass_state_to_policy,
                 pass_latent_to_policy,
                 dim_state,
                 dim_latent,
                 # hidden
                 hidden_layers,
                 activation_function,  # tanh, relu, leaky-relu
                 policy_initialisation,  # orthogonal / normc
                 # output
                 action_space,
                 init_std,
                 pass_task_context_to_policy=False,
                 dim_task_context=None,
                 context_activation_function=None,
                 context_hidden_layers=None,
                 # normalisation
                 norm_state=False,
                 norm_latent=False,
                 norm_context=False,
                 ):
        """
        The policy can get any of these as input:
        - state (given by environment)
        - latent variable (from encoder)
        """
        super(HyperPolicy, self).__init__()

        self.args = args

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError(f'Got unexpected activation function for policy: {activation_function}')

        if policy_initialisation == 'normc':
            init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0),
                                   nn.init.calculate_gain(activation_function))
        elif policy_initialisation == 'orthogonal':
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                                   nn.init.calculate_gain(activation_function))
        else:
            if activation_function == 'tanh' or activation_function == 'relu':
                raise ValueError('You should use policy initialisation')
            init_ = None

        self.pass_state_to_policy = pass_state_to_policy
        self.pass_latent_to_policy = pass_latent_to_policy
        self.pass_task_context_to_policy = pass_task_context_to_policy

        # set normalisation parameters for the inputs
        # (will be updated from outside using the RL batches)
        self.norm_state = norm_state
        assert not self.norm_state or (self.norm_state and dim_state is not None)
        self.norm_latent = norm_latent and (dim_latent is not None)
        if self.pass_latent_to_policy and self.norm_latent:
            self.latent_rms = TorchRunningMeanStd(shape=(dim_latent))
        self.norm_task_context = norm_context and (dim_task_context is not None)
        if self.pass_task_context_to_policy and self.norm_task_context:
            # FIXME
            raise NotImplementedError

        # FIXME Use a hypernetwork for the online latent too
        curr_input_dim = dim_state * int(self.pass_state_to_policy) \
                         + dim_latent * int(self.pass_latent_to_policy)

        # initialise encoders for separate inputs
        self.use_state_encoder = self.args.policy_state_embedding_dim is not None
        if self.pass_state_to_policy and self.use_state_encoder:
            self.state_encoder = utl.FeatureExtractor(dim_state,
                                                      self.args.policy_state_embedding_dim,
                                                      self.activation_function)
            curr_input_dim = curr_input_dim - dim_state + self.args.policy_state_embedding_dim

        self.use_latent_encoder = self.args.policy_latent_embedding_dim is not None
        if self.pass_latent_to_policy and self.use_latent_encoder:
            self.latent_encoder = utl.FeatureExtractor(dim_latent,
                                                       self.args.policy_latent_embedding_dim,
                                                       self.activation_function)
            curr_input_dim = curr_input_dim - dim_latent + self.args.policy_latent_embedding_dim

        # initialise actor and critic (static policy network)
        hidden_layers = [int(h) for h in hidden_layers]
        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            if init_ is not None:
                fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
            else:
                fc = nn.Linear(curr_input_dim, hidden_layers[i])
            self.actor_layers.append(fc)
            if init_ is not None:
                fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
            else:
                fc = nn.Linear(curr_input_dim, hidden_layers[i])
            self.critic_layers.append(fc)
            curr_input_dim = hidden_layers[i]
        self.critic_linear = nn.Linear(hidden_layers[-1], 1)

        self.use_task_context_encoder = (self.args.policy_task_context_embedding_dim is not None
                                         and self.args.policy_task_context_embedding_dim > 0)
        context_embedding_dim = self.args.policy_task_context_embedding_dim if self.use_task_context_encoder \
            else dim_task_context

        if self.pass_task_context_to_policy:
            assert context_activation_function is not None
            if context_activation_function == 'tanh':
                self.context_activation_function = nn.Tanh()
            elif context_activation_function == 'relu':
                self.context_activation_function = nn.ReLU()
            elif context_activation_function == 'leaky-relu':
                self.context_activation_function = nn.LeakyReLU()
            elif context_activation_function == 'selu':
                self.context_activation_function = nn.SELU()
            else:
                raise ValueError(f'Got unexpected activation function for policy context: '
                                 f'{context_activation_function}')

            if self.use_task_context_encoder:
                if context_hidden_layers is None or len(context_hidden_layers) == 0:
                    self.task_context_encoder = utl.FeatureExtractor(dim_task_context,
                                                                     context_embedding_dim,
                                                                     activation_function=self.context_activation_function)
                else:
                    self.task_context_encoder = MLPModule(dim_task_context,
                                                          output_dim=context_embedding_dim,
                                                          hidden_sizes=context_hidden_layers,
                                                          hidden_nonlinearity=self.context_activation_function,
                                                          output_nonlinearity=self.context_activation_function, )

        hypernet_args = dict(
            target_shapes=[
                [hidden_layers[-1], 32],  # layer 1
                [32],  # bias 1
                [32, hidden_layers[-1]],  # layer 2
                [hidden_layers[-1]],  # bias 2
            ],
            uncond_in_size=0,
            cond_in_size=context_embedding_dim,
            layers=self.args.hypernetwork_layers,
            activation_fn=self.context_activation_function,
            use_bias=True,
            chunk_size=self.args.hypernetwork_chunk_size,
        )

        self.actor_context_hypernet = ChunkedHMLP(**hypernet_args)
        self.critic_context_hypernet = ChunkedHMLP(**hypernet_args)

        self.actor_context_hypernet.apply_chunked_hyperfan_init()
        self.critic_context_hypernet.apply_chunked_hyperfan_init()

        # output distributions of the policy
        self.action_encoder = None
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(hidden_layers[-1], num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(hidden_layers[-1], num_outputs, init_std, self.args.norm_actions_pre_sampling)
        elif (action_space.__class__.__name__ == "OneHotEncoding" or
              action_space.__class__.__name__ == "TorchOneHotEncoding"):
            # FIXME Make sure only MEWA uses this, not Meta-World
            self.num_outputs = int(np.prod(action_space.shape))
            self.dist = Categorical(hidden_layers[-1], self.num_outputs)
            self.action_encoder = nn.functional.one_hot
        else:
            raise NotImplementedError

    def forward_actor(self, inputs, task_context, use_pre_activations):
        h = inputs
        for i in range(len(self.actor_layers)):
            h = self.actor_layers[i](h)
            if use_pre_activations and i == len(self.actor_layers) - 1:
                pre_activations = h.clone()
            if i != len((self.actor_layers)) - 1:
                # Do not use an activation before the adapter
                h = self.activation_function(h)

        params = self.actor_context_hypernet.forward(cond_input=task_context, ret_format='sequential')
        h = forward_target_mlp(h, params, self.activation_function, task_context.shape,
                               return_pre_activations=use_pre_activations)

        if use_pre_activations:
            # Also return the last value at the penultimate layer (i.e. last layer of the hypernetwork)
            h, pre_activations = h
            return h, pre_activations
        return h

    def forward_critic(self, inputs, task_context):
        h = inputs
        for i in range(len(self.critic_layers)):
            h = self.critic_layers[i](h)
            h = self.activation_function(h)

        params = self.critic_context_hypernet.forward(cond_input=task_context, ret_format='sequential')
        h = forward_target_mlp(h, params, self.activation_function, task_context.shape)
        return h

    def forward(self, state, latent, task_context=None, use_pre_activations=False):
        # handle inputs (normalise + embed)
        if self.pass_state_to_policy:
            if self.norm_state:
                state = (state - utl.state_rms().mean) / torch.sqrt(utl.state_rms().var + 1e-8)
            if self.use_state_encoder:
                state = self.state_encoder(state)
        else:
            state = torch.zeros(0, ).to(global_device())

        if self.pass_latent_to_policy:
            if self.norm_latent:
                latent = (latent - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)
            if self.use_latent_encoder:
                latent = self.latent_encoder(latent)
            # FIXME Find a better way of doing this when num_processes == 1
            if len(latent.shape) == 1:
                latent = latent.unsqueeze(0)

        if self.pass_task_context_to_policy:
            if self.norm_task_context:
                raise NotImplementedError
            if self.use_task_context_encoder:
                task_context = self.task_context_encoder(task_context)  # (p, d_ctx)

        # concatenate inputs
        inputs = [state]
        if self.pass_latent_to_policy:
            inputs.append(latent)
        inputs = torch.cat(inputs, dim=-1)

        # forward through critic/actor part
        hidden_critic = self.critic_linear(self.forward_critic(inputs, task_context))
        hidden_actor = self.forward_actor(inputs, task_context, use_pre_activations=use_pre_activations)  # (p, q, m*H, 128)

        if use_pre_activations:
            hidden_actor, pre_activations = hidden_actor
            return hidden_critic, hidden_actor, pre_activations
        return hidden_critic, hidden_actor

    def act(self, state, latent, task_context=None, deterministic=False):
        """
        Returns the (raw) actions and their value.
        """
        value, actor_features = self.forward(state=state, latent=latent, task_context=task_context)
        dist = self.dist(actor_features)
        if deterministic:
            if isinstance(dist, FixedCategorical):
                action = dist.mode()
            else:
                action = dist.mean
        else:
            action = dist.sample()

        if self.action_encoder == nn.functional.one_hot:
            # The action encoder will add an extra dimension to the action. If this results in a 3D tensor, remove the
            # extra dimension
            action = self.action_encoder(action, num_classes=self.num_outputs)
            while len(action.shape) > 2:
                action = action.squeeze(1)

        return value, action

    def get_value(self, state, latent):
        value, _ = self.forward(state, latent)
        return value

    def evaluate_actions(self, state, latent, action, task_context=None, use_pre_activations=False, return_dist=False):
        if not use_pre_activations:
            value, actor_features = self.forward(state, latent, task_context)  # (), (p, q, m*H, 128)
        else:
            value, actor_features, pre_activations = self.forward(state, latent, task_context, use_pre_activations=True)
        dist = self.dist.forward(actor_features)  # (p, q, m*H, |A|)

        if self.args.norm_actions_post_sampling:
            raise NotImplementedError
        else:
            # Decode the one-hot action vector
            if self.action_encoder == nn.functional.one_hot:
                action = torch.argmax(action, dim=-1).unsqueeze(-1)  # (p, q, m*H, 1)

            action_log_probs = dist.log_probs(action)  # (p, q, m*H, 1)
            dist_entropy = dist.entropy().mean()  # (0)

        if use_pre_activations:
            return value, action_log_probs, dist_entropy, pre_activations
        if return_dist:
            return value, action_log_probs, dist_entropy, dist
        return value, action_log_probs, dist_entropy