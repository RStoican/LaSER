from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from laser.garage.torch.modules import TransformerEncoderLayerNoLN, PositionalEncoding
from laser.garage.utils.running_stats import RunningMeanStd


class TransformerEncoder(nn.Module, ABC):
    """Transformer whose outputs are fed into a Normal distribution.

    A policy that contains a Transformer to make prediction based on a gaussian distribution.

    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        latent_layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.
    """

    def __init__(self,
                 env_spec,
                 action_dim,
                 name='MetaRLTransformerEncoder',

                 num_encoder_layers=6,
                 nhead=8,
                 d_model=128,
                 dim_feedforward=512,
                 transformer_encoder_activation='gelu',
                 tfixup=True,
                 remove_ln=True,
                 normalize_wm=False,
                 final_transformer_projection=False,

                 out_latent_dim=32,
                 multiheaded_latent_network=False,
                 shared_latent_pooling_type='max',
                 simple_shared_latent=False,
                 task_latent_type='diagonal',
                 task_latent_multidim=False,
                 latent_hidden_sizes=(64,),
                 latent_hidden_nonlinearity=torch.tanh,
                 latent_hidden_w_init=nn.init.xavier_uniform_,
                 latent_hidden_b_init=nn.init.zeros_,
                 latent_output_nonlinearity=None,
                 latent_output_b_init=nn.init.zeros_,
                 latent_layer_normalization=False,

                 traj_per_meta_traj=10,
                 max_trajectory_len=100,
                 meta_train_tasks=200,
                 meta_train_meta_trajs=30,

                 dropout=0.0,
                 train_mode=True,
                 ):
        super().__init__()
        self._env_spec = env_spec
        self._name = name

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = action_dim
        self._step_size = self._obs_dim + self._action_dim + 1  # ((obs_t, done_t), act_t, rew_t)
        self._traj_per_meta_traj = traj_per_meta_traj  # number of trajectories per meta-trajectory (i.e. m)
        self._traj_len = max_trajectory_len  # number of steps in a single trajectory (i.e. horizon H)
        self._meta_traj_len = self._traj_per_meta_traj * self._traj_len  # m*H
        self._d_model = d_model
        self._normalize_wm = normalize_wm
        self._out_latent_dim = out_latent_dim
        self._multiheaded_latent_network = multiheaded_latent_network
        self._task_latent_type = task_latent_type
        self._task_latent_multidim = task_latent_multidim
        self._final_transformer_projection = final_transformer_projection

        self._latent_hidden_sizes = latent_hidden_sizes
        self._latent_hidden_nonlinearity = latent_hidden_nonlinearity
        if isinstance(self._latent_hidden_nonlinearity, str):
            self._latent_hidden_nonlinearity = getattr(F, self._latent_hidden_nonlinearity)
        self._latent_hidden_w_init = latent_hidden_w_init
        self._latent_hidden_b_init = latent_hidden_b_init
        self._latent_output_nonlinearity = latent_output_nonlinearity
        if isinstance(self._latent_output_nonlinearity, str):
            self._latent_output_nonlinearity = getattr(F, self._latent_output_nonlinearity)
        self._latent_output_b_init = latent_output_b_init
        self._latent_layer_normalization = latent_layer_normalization
        self._train_mode = train_mode

        assert self._task_latent_type in ['diagonal', 'orthogonal', 'attention'], f'Got {self._task_latent_type}'

        # During meta-training, we will have enough tasks and meta-trajectories to learn the different components of
        # the latent space. So, set how many tasks and meta-trajectories we use for that
        # The number of tasks used during meta-training (i.e. p)
        self._meta_train_tasks = meta_train_tasks
        # the number of meta-trajectories per task used during meta-training (i.e. q)
        self._meta_train_meta_trajs = meta_train_meta_trajs

        self._shared_latent_pooling_type = shared_latent_pooling_type
        if self._shared_latent_pooling_type == 'none' or self._shared_latent_pooling_type == 'None':
            self._shared_latent_pooling_type = None
        self._simple_shared_latent = simple_shared_latent

        self._step_embedding = nn.Linear(
            in_features=self._step_size,
            out_features=self._d_model,
            bias=False
        )

        self._positional_encoding = PositionalEncoding(
            d_model=self._d_model,
            dropout=dropout,
            max_len=self._meta_traj_len,
        )

        # Create the encoder layers, depending on whether we use Layer Normalisation or not
        encoder_layer_type = TransformerEncoderLayerNoLN if remove_ln else nn.TransformerEncoderLayer
        encoder_layers = encoder_layer_type(
            d_model=self._d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=transformer_encoder_activation
        )

        self._transformer_module = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        for p in self._transformer_module.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        if tfixup:
            self._do_tfixup(self._d_model, num_encoder_layers)

        # The latent heads are the output of the encoder. The particular heads used depends on the encoder type
        self._shared_latent_head, self._task_latent_head, self._traj_latent_head = None, None, None
        self._shared_latent_pooling = None
        self._create_latent_heads()

        self.src_mask = None

        if self._normalize_wm:
            self.wm_rms = RunningMeanStd(shape=(self._obs_dim,))

        self._prev_observations = None
        self._last_hidden_state = None
        self._prev_actions = None
        self._episodic_memory_counter = None
        self._new_episode = None
        self._step = None

        assert self._is_valid_encoder(), f'{self._shared_latent_head}\n{self._task_latent_head}\n' \
                                         f'{self._traj_latent_head}\n{self._shared_latent_pooling}\n'

    @abstractmethod
    def _create_latent_heads(self):
        """"""

    @abstractmethod
    def _compute_latent(self, transformer_output, compute_shared_latent=True, compute_task_latent=True):
        """"""

    @abstractmethod
    def _get_attention_mask(self, seq_len=None):
        """"""

    @abstractmethod
    def _is_valid_encoder(self):
        """"""

    def forward(self, trajectories,
                compute_shared_latent=True, compute_task_latent=True, compute_per_episode=False):
        """Compute the hidden memory state. Then, compute a distribution over latent task identifiers using the memory

                Args:
                    :param trajectories: (torch.Tensor or tuple or list) Batch of trajectories (obs, act, rew)
                    on default torch device.
                        obs+done shape (p, q, obs_dim+1, m*H)
                        act shape (p, q, act_dim, m*H)
                        rew shape (p, q, 1, m*H)
                    where
                        p is the number of tasks,
                        q is the number of meta-trajectories per task,
                        m is the number of trajectories per meta-trajectory,
                        H is the horizon (length) of a trajectory
                        obs_dim, act_dim are the dimensions of the observations and actions, repsectively
                    :param compute_latent_for_each_timestep: (bool) By default (False), only compute the distribution for the
                        last timestep H of each meta-trajectory given. Otherwise, compute a distribution over each timestep t,
                        using only the previous [0, 1, ..., t] timesteps, ignoring future timesteps [t+1, t+2, ..., H]

                Returns:
                    :return all_latent_samples: a sample from the distribution
                    :return all_latent_means: the mean of the distribution
                    :return all_latent_logvars: the log std of the distribution
                    :return latent_memory_state: the hidden memory state used to compute the distribution

                """

        if isinstance(trajectories, tuple) or isinstance(trajectories, list):
            trajectories = torch.cat((trajectories[0], trajectories[1], trajectories[2]), dim=-1)  # (p, q, dim, m*H)
        if not isinstance(trajectories, torch.Tensor):
            raise ValueError('Transformer Encoder: The trajectories to encode should be a tuple, list or torch.Tensor '
                             'of (obs, act, rew)')

        if compute_per_episode:
            num_tasks = trajectories.shape[0]  # p
            num_meta_traj = trajectories.shape[1]  # q
            trajectories = trajectories.reshape(num_tasks, num_meta_traj, self._step_size,
                                                self._traj_per_meta_traj, self._traj_len)  # (p, q, dim, m, H)
            trajectories = trajectories.permute(0, 1, 3, 2, 4)  # (p, q, m, dim, H)
            trajectories = trajectories.reshape(num_tasks, -1, self._step_size, self._traj_len)  # (p, q*m, dim, H)

        transformer_output = self._compute_memories(trajectories)  # (p, q, m*H, d_model)

        if compute_per_episode:
            transformer_output = transformer_output.reshape(num_tasks, num_meta_traj,
                                                            -1, self._d_model)  # (p, q, m*H, d_model)

        # Compute the latent, depending on whether we are using a unidirectional or bidirectional encoder
        #   Unidirectional: trajectory latent  (p, q, H*d_z, m)
        #   Bidirectional: shared latent and task latent  (H*d_z), (p, q, H*d_z)
        return self._compute_latent(transformer_output,
                                    compute_shared_latent,
                                    compute_task_latent)

    def _compute_memories(self, trajectories):
        # Reshape into a single batch dimension, each containing different meta-trajectories (from either the same or
        # different tasks)
        num_tasks = trajectories.shape[0]  # p
        num_meta_traj = trajectories.shape[1]  # q
        batch_size = num_tasks * num_meta_traj  # batch = p*q
        trajectories = trajectories.reshape(batch_size, self._step_size, -1)  # (batch, dim, m*H)
        trajectories = trajectories.permute(0, 2, 1)  # (batch, m*H, dim)

        # Computing working memory as a representation from tuple (obs, act, rew)
        working_memo = self._step_embedding(trajectories)  # (batch, m*H, d_model)

        # Positional encoding
        working_memo = working_memo.permute(1, 0, 2)  # Transformer module inputs (m*H, batch, d_model)
        wm_pos = self._positional_encoding.forward(working_memo)  # (m*H, batch, d_model)

        # Transformer encoder. Compute either a unidirectional or bidirectional output, depending on the masking
        attention_mask = self._get_attention_mask(seq_len=wm_pos.shape[0])
        transformer_output = self._transformer_module(wm_pos, mask=attention_mask)  # (m*H, batch, d_model)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch, m*H, d_model)
        transformer_output = transformer_output.reshape(num_tasks, num_meta_traj, -1,
                                                        self._d_model)  # (p, q, m*H, d_model)

        if self._final_transformer_projection:
            raise NotImplementedError
            transformer_output = self._final_transformer_layer(transformer_output)  # (p, q, m*H, d_z)

        return transformer_output

    def mlp_output_w_init(self, x):
        # FIXME Use args.output_weights_scale
        return torch.nn.init.xavier_uniform_(x, gain=0.01)  # self.args.output_weights_scale)

    def _do_tfixup(self, d_model, num_encoder_layers):
        for p in self._step_embedding.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, 0, d_model ** (- 1. / 2.))

        temp_state_dic = {}
        for name, param in self._step_embedding.named_parameters():
            if 'weight' in name:
                temp_state_dic[name] = ((9 * num_encoder_layers) ** (- 1. / 4.)) * param

        for name in self._step_embedding.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self._step_embedding.state_dict()[name]
        self._step_embedding.load_state_dict(temp_state_dic)

        temp_state_dic = {}
        for name, param in self._transformer_module.named_parameters():
            if any(s in name for s in ["linear1.weight", "linear2.weight", "self_attn.out_proj.weight"]):
                temp_state_dic[name] = (0.67 * num_encoder_layers ** (- 1. / 4.)) * param
            elif "self_attn.in_proj_weight" in name:
                temp_state_dic[name] = (0.67 * num_encoder_layers ** (- 1. / 4.)) * (param * (2 ** 0.5))

        for name in self._transformer_module.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self._transformer_module.state_dict()[name]
        self._transformer_module.load_state_dict(temp_state_dic)

    @property
    def name(self):
        """
        Returns:
            str: Name of encoder
        """
        return self._name

    @property
    def task_latent_type(self):
        return self._task_latent_type
