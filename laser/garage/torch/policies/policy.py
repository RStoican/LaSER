"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import numpy as np
import torch
import torch.nn as nn

from laser.garage.torch._functions import global_device
from laser.garage.torch.modules.mlp_module import MLPModule
from laser.garage.utils import helpers as utl
from laser.garage.utils.running_stats import TorchRunningMeanStd

try:
    from torch.distributions import TanhTransform, TransformedDistribution


    class TanhNormal(TransformedDistribution):
        def __init__(self, base_distribution, transforms, validate_args=None):
            super().__init__(base_distribution, transforms, validate_args=None)


    @property
    def mean(self):
        x = self.base_dist.mean
        for transform in self.transforms:
            x = transform(x)
        return x

except ImportError:
    print('You are probably running MuJoCo 131, so PyTorch Transforms cannot be used. '
          'Do not set norm_actions_pre_sampling, this will break.')
    pass


class Policy(nn.Module):
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
        super(Policy, self).__init__()

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
            self.context_rms = TorchRunningMeanStd(shape=dim_task_context)

        curr_input_dim = dim_state * int(self.pass_state_to_policy) \
                         + dim_latent * int(self.pass_latent_to_policy) \
                         + (0 if dim_task_context is None else dim_task_context) * int(self.pass_task_context_to_policy)
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
        self.use_task_context_encoder = (self.args.policy_task_context_embedding_dim is not None
                                         and self.args.policy_task_context_embedding_dim > 0)
        if self.pass_task_context_to_policy and self.use_task_context_encoder:
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

            if context_hidden_layers is None:
                self.task_context_encoder = utl.FeatureExtractor(dim_task_context,
                                                                 self.args.policy_task_context_embedding_dim,
                                                                 activation_function=self.context_activation_function)
            else:
                self.task_context_encoder = MLPModule(dim_task_context,
                                                      output_dim=self.args.policy_task_context_embedding_dim,
                                                      hidden_sizes=context_hidden_layers,
                                                      hidden_nonlinearity=self.context_activation_function,
                                                      output_nonlinearity=self.context_activation_function, )
            curr_input_dim = curr_input_dim - dim_task_context + self.args.policy_task_context_embedding_dim

        # initialise actor and critic
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

    def actor_params(self):
        return [self.actor_layers.parameters(), self.dist.parameters()]

    def critic_params(self):
        return [self.critic_layers.parameters(), self.critic_linear.parameters()]

    def encoder_params(self):
        encoder_params = []
        if self.pass_state_to_policy and self.use_state_encoder:
            encoder_params.append(self.state_encoder.parameters())
        if self.pass_latent_to_policy and self.use_latent_encoder:
            encoder_params.append(self.latent_encoder.parameters())
        if self.pass_task_context_to_policy and self.use_task_context_encoder:
            encoder_params.append(self.task_context_encoder.parameters())
        return encoder_params

    def forward_actor(self, inputs, use_pre_activations):
        h = inputs
        for i in range(len(self.actor_layers)):
            h = self.actor_layers[i](h)
            if use_pre_activations and i == len((self.actor_layers)) - 1:
                pre_activations = h.clone()
            h = self.activation_function(h)
        if use_pre_activations:
            return h, pre_activations
        return h

    def forward_critic(self, inputs):
        h = inputs
        for i in range(len(self.critic_layers)):
            h = self.critic_layers[i](h)
            h = self.activation_function(h)
        return h

    def encode_task_context(self, task_context, state_shape):
        if self.norm_task_context:
            task_context = (task_context - self.context_rms.mean) / torch.sqrt(self.context_rms.var + 1e-8)
        if self.use_task_context_encoder:
            task_context = self.task_context_encoder(task_context)  # (p, d_ctx)
        if 2 < len(state_shape) != len(task_context.shape):
            # Full trajectory case: the same context is added to each step, so duplicate it
            horizon = state_shape[-2]  # H
            task_context = task_context.unsqueeze(-2).repeat(1, 1, horizon, 1)  # (p, q, H, d_ctx)
        return task_context

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
            task_context = self.encode_task_context(task_context, state.shape)

        # FIXME Find a better way of doing this when num_processes == 1
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # concatenate inputs
        inputs = [state]
        if self.pass_latent_to_policy:
            inputs.append(latent)
        if self.pass_task_context_to_policy:
            inputs.append(task_context)
        inputs = torch.cat(inputs, dim=-1)

        # forward through critic/actor part
        hidden_critic = self.critic_linear(self.forward_critic(inputs))
        hidden_actor = self.forward_actor(inputs, use_pre_activations=use_pre_activations)  # (p, q, m*H, 128)

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

    def update_rms(self, args, policy_storage):
        """ Update normalisation parameters for inputs with current data """
        if self.pass_latent_to_policy and self.norm_latent:
            # FIXME The latent rms is computed from padding steps
            # raise NotImplementedError('The latent rms will be computed from padding steps')
            latent = utl.get_latent_for_policy(args, policy_storage.latent_traj)
            self.latent_rms.update(latent)
        if self.pass_task_context_to_policy and self.norm_task_context:
            self.context_rms.update(policy_storage.task_context.unsqueeze(-1))

    def evaluate_actions(self, state, latent, action, task_context=None, use_pre_activations=False, return_dist=False):
        if not use_pre_activations:
            value, actor_features = self.forward(state, latent, task_context)  # (), (p, q, m*H, 128)
        else:
            value, actor_features, pre_activations = self.forward(state, latent, task_context, use_pre_activations=True)
        dist = self.dist.forward(actor_features)  # (p, q, m*H, |A|)

        if self.args.norm_actions_post_sampling:
            raise NotImplementedError
            transformation = TanhTransform(cache_size=1)
            dist = TanhNormal(dist, transformation)
            action = transformation(action)
            action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
            # empirical entropy
            # dist_entropy = -action_log_probs.mean()
            # entropy of underlying dist (isn't correct but works well in practice)
            dist_entropy = dist.base_dist.entropy().mean()
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


FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)  # (p, q, m*H, |A|)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std, norm_actions_pre_sampling):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(np.log(torch.zeros(num_outputs) + init_std))
        self.norm_actions_pre_sampling = norm_actions_pre_sampling
        self.min_std = torch.tensor([1e-6]).to(global_device())

    def forward(self, x):
        action_mean = self.fc_mean(x)
        if self.norm_actions_pre_sampling:
            action_mean = torch.tanh(action_mean)
        std = torch.max(self.min_std, self.logstd.exp())
        dist = FixedNormal(action_mean, std)

        return dist


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().reshape(1, -1)
        else:
            bias = self._bias.t().reshape(1, -1, 1, 1)

        return x + bias
