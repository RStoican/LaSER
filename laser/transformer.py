from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn as nn

from laser.garage.storage.dataset_storage import DatasetStorage, DatasetStoragePrecollect
from laser.garage.torch._functions import global_device
from laser.garage.trainers.transformer_loss_computer import TransformerLossComputer
from laser.garage.torch.modules import MultiInputMLPModule, ReconstructionModule
from laser.garage.torch.modules.attention_encoder_layer import AttentionEncoderLayer
from laser.garage.torch.modules.mlp_module import MultiHeadedMLPModule, MultiInputMultiHeadedMLPModule
from laser.garage.utils import helpers as utl
from laser.garage.utils.helpers import step_to_traj, traj_to_step


class Transformer(nn.Module):
    def __init__(self, args, uni_encoder, bi_encoder, logger, get_iter_idx,
                 env_spec,
                 dataset_size,
                 action_dim,
                 lr,
                 out_latent_dim=32,
                 mlp_hidden_nonlinearity=F.relu,
                 d_model=128,
                 traj_per_meta_traj=10,
                 max_trajectory_len=100,
                 meta_train_tasks=200,
                 meta_train_meta_trajs=30,
                 reconstruction_hidden_sizes=(64, 64),
                 linear_coeff_hidden_sizes=(64, 64),
                 masked_reconstruction_loss=False,
                 use_traj_latent_for_context=True,
                 norm_state=True,
                 train_mode=True,
                 finetune_mode=True,
                 dataset=None,
                 store_norm_rewards=False,
                 ):
        super().__init__()
        self.args = args
        self.logger = logger
        self.get_iter_idx = get_iter_idx

        self.uni_encoder = uni_encoder
        self.bi_encoder = bi_encoder

        # Use the rollout storage to pre-train a transformer with off-policy data, using the reconstruction loss.
        if dataset is None or not dataset['use_starting_dataset']:
            self.dataset_storage = DatasetStorage(num_processes=self.args.num_processes,
                                                  dataset_size=dataset_size,
                                                  meta_traj_per_task=meta_train_meta_trajs,
                                                  traj_per_meta_traj=self.args.traj_per_meta_traj,
                                                  max_trajectory_len=self.args.horizon,
                                                  obs_dim=self.args.state_dim,
                                                  action_dim=self.args.action_dim,
                                                  store_norm_rewards=store_norm_rewards,
                                                  )
        else:
            self.dataset_storage = DatasetStoragePrecollect(num_processes=self.args.num_processes,
                                                            dataset_size=dataset_size,
                                                            meta_traj_per_task=meta_train_meta_trajs,
                                                            traj_per_meta_traj=self.args.traj_per_meta_traj,
                                                            max_trajectory_len=self.args.horizon,
                                                            obs_dim=self.args.state_dim,
                                                            action_dim=self.args.action_dim,
                                                            dataset_path=dataset['path'],
                                                            starting_len=dataset['len'],
                                                            )

        # The shared latent is only computed when we are training the encoder, and the input contains many tasks.
        # When we compute the latent space to be used by a policy (i.e. the input has 1 or a low number of tasks), we
        # use a fixed shared latent (i.e. previously computed by the encoder from many tasks)
        self._static_shared_latent = None

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = action_dim
        self._lr = lr
        self._step_size = self._obs_dim + self._action_dim + 1  # ((obs_t, done_t), act_t, rew_t)
        self._d_model = d_model
        self._out_latent_dim = out_latent_dim
        self._traj_per_meta_traj = traj_per_meta_traj  # number of trajectories per meta-trajectory (i.e. m)
        self._traj_len = max_trajectory_len  # number of steps in a single trajectory (i.e. horizon H)
        self._meta_traj_len = self._traj_per_meta_traj * self._traj_len  # m*H
        self._task_latent_type = self.uni_encoder.task_latent_type
        assert self._task_latent_type == self.bi_encoder.task_latent_type
        self._task_latent_multidim = self.bi_encoder._task_latent_multidim
        self._use_traj_latent_for_context = use_traj_latent_for_context
        self._norm_state = norm_state

        # During meta-training, we will have enough tasks and meta-trajectories to learn the different components of
        # the latent space. So, set how many tasks and meta-trajectories we use for that
        # The number of tasks used during meta-training (i.e. p)
        self._meta_train_tasks = meta_train_tasks
        # the number of meta-trajectories per task used during meta-training (i.e. q)
        self._meta_train_meta_trajs = meta_train_meta_trajs

        if self._task_latent_type == 'attention':
            self._traj_embedding = nn.Linear(
                in_features=self._d_model * self._traj_len,
                out_features=self._d_model,
                bias=False
            )
            if not self.args.simple_shared_latent:
                self._shared_in = self._d_model * self._traj_len
            else:
                self._shared_in = self._d_model
            self._shared_embedding = nn.Linear(
                in_features=self._shared_in,
                out_features=self._d_model,
                bias=False
            )

            self._task_attention = AttentionEncoderLayer(
                d_model=d_model,
                nhead=self.args.n_heads,
                dim_feedforward=self.args.dim_ff,
                dropout=self.args.dropout,
                activation=self.args.transformer_encoder_activation,
                # FIXME
                use_residual=True,
            )
            self._context_attention = AttentionEncoderLayer(
                d_model=d_model,
                nhead=self.args.n_heads,
                dim_feedforward=self.args.dim_ff,
                dropout=self.args.dropout,
                activation=self.args.transformer_encoder_activation,
                # FIXME
                use_residual=True,
            )

        self.train_mode = train_mode
        if self.train_mode:
            # The unembedding MLP head will convert the transformer output to a tensor of the same size as the input.
            # This head is used only for self-supervised training for the transformer encoder. The goal is to learn to
            # reconstruct the input from the transformer's output
            self._unembedding_head = ReconstructionModule(
                input_dim=self._d_model,
                obs_dim=self._obs_dim,
                act_dim=self._action_dim,
                rew_dim=1,
                hidden_sizes=reconstruction_hidden_sizes,
                hidden_nonlinearity=mlp_hidden_nonlinearity,
            )

            task_latent_input_dim = self._meta_train_meta_trajs * self._out_latent_dim * self._traj_len  # (q*H*d_z)
            if self._task_latent_type == 'attention':
                task_latent_input_dim = self._meta_train_meta_trajs * self._out_latent_dim  # (q*d_z)
                if self._task_latent_multidim:
                    task_latent_input_dim *= self._meta_traj_len  # (q*m*H*d_z)

            traj_latent_input_dim = self._out_latent_dim * self._meta_traj_len  # (d_z*m*H)
            if self.args.no_traj_latent_for_task_rec:
                traj_latent_input_dim \
                    = int(task_latent_input_dim / self._meta_train_meta_trajs)  # (H*d_z) / (d_z) / (m*H*d_z)

            output_dim = self._meta_train_meta_trajs * self._traj_per_meta_traj * self._traj_per_meta_traj  # (q*m*m)

            self._linear_coeff_head = MultiInputMLPModule(
                n_inputs=2,
                input_dims=(
                    task_latent_input_dim,
                    traj_latent_input_dim
                ),  # (q*m*H*d_z), (d_z*m*H)
                output_dim=output_dim,  # (q*m*m)
                hidden_sizes=linear_coeff_hidden_sizes,
                hidden_nonlinearity=mlp_hidden_nonlinearity,
                input_nonlinearity=mlp_hidden_nonlinearity,
            )
            if self.args.task_reconstruction_type == 'linear':
                pass
            elif self.args.task_reconstruction_type == 'non-linear':
                self._w_size = self.args.task_reconstruction_latent_size
                multi_headed_module = MultiHeadedMLPModule(
                    n_heads=4,
                    input_dim=output_dim,  # (q*m*m)
                    output_dims=(
                        self._meta_train_meta_trajs * self._traj_per_meta_traj * self._w_size,
                        self._meta_train_meta_trajs * self._w_size,
                        self._meta_train_meta_trajs * self._traj_per_meta_traj * self._w_size,
                        self._meta_train_meta_trajs * self._traj_per_meta_traj
                    ),  # (q*m*w), (q*w), (q*w*m), (q*m)
                    hidden_sizes=[linear_coeff_hidden_sizes[-1]],
                    hidden_nonlinearity=mlp_hidden_nonlinearity,
                    output_nonlinearities=None,
                )
                self._linear_coeff_head = MultiInputMultiHeadedMLPModule(
                    multi_input_module=self._linear_coeff_head,
                    multi_headed_module=multi_headed_module,
                )
            else:
                raise ValueError(
                    f'Expected task_reconstruction_type to be: linear or non-linear. Got {self.args.task_reconstruction_type}')

            self._loss_computer = TransformerLossComputer(args,
                                                          masked_reconstruction_loss=masked_reconstruction_loss,
                                                          task_latent_type=self._task_latent_type,
                                                          traj_per_meta_traj=self._traj_per_meta_traj,
                                                          traj_len=self._traj_len,
                                                          obs_dim=self._obs_dim, act_dim=self._action_dim, )

        if self.train_mode or finetune_mode:
            # TODO Look into the optimiser used in Transformer MetaRL
            self._optimiser = self.initialise_optimiser()

    def initialise_optimiser(self):
        return torch.optim.Adam([*self.parameters()], lr=self._lr)

    def get_headless_state_dict(self):
        if not self.train_mode:
            return self.state_dict()

        head_prefixes = ['_unembedding_head', '_linear_coeff_head', 'bi_encoder._shared_latent_head']
        state_dict = self.state_dict()

        headless_state_dict = {
            key: value for key, value in state_dict.items()
            if not key.startswith(tuple(head_prefixes))
        }

        headless_state_dict = OrderedDict(headless_state_dict)
        return headless_state_dict

    def forward_exploration(self, trajectories=None, return_latest_latent=True,
                            prior=False, num_tasks=None, starting_state=None):
        assert (trajectories is not None) != (prior and num_tasks is not None)

        if prior:
            trajectories = self._create_prior_trajectory(num_tasks, starting_state)  # (p, 1, d, m*H)

        _, _, _, traj_latent = self.forward(trajectories,
                                            compute_shared_latent=False,
                                            compute_task_latent=False, )  # (p, 1, H*d_z, m)

        # Convert from trajectories to steps. This will inverse the conversion done by the unidirectional encoder
        traj_latent = traj_to_step(traj_latent, self._traj_len)  # (p, 1, d_z, m*H)

        if not return_latest_latent:
            return traj_latent.squeeze(1)  # (p, d_z, m*H)

        # Keep only the latent of the latest (unpadded) step
        curr_em_index = self._compute_memory_index(trajectories)  # (p, q)
        curr_em_index = curr_em_index.unsqueeze(-1).repeat(1, 1, self._out_latent_dim).unsqueeze(-1)  # (p, 1, d_z, 1)
        latest_latent = torch.gather(traj_latent, dim=-1, index=curr_em_index).squeeze(-1)  # (p, 1, d_z)
        return latest_latent.squeeze(1)  # (p, d_z)

    def forward(self, trajectories,
                compute_shared_latent=True, compute_task_latent=True, compute_traj_latent=True,
                return_reconstructed_input=False, use_static_shared_latent=False):
        trajectories = self._do_state_norm(trajectories) if self._norm_state else trajectories

        shared_latent, task_latent, traj_latent = None, None, None
        if compute_traj_latent:
            traj_latent = self.uni_encoder(trajectories, compute_per_episode=self.args.individual_exploration_episodes)  # (p, q, H*d_z, m)
        if compute_shared_latent or compute_task_latent:
            shared_latent, task_latent = self.bi_encoder(trajectories,
                                                         compute_shared_latent,
                                                         compute_task_latent)  # (d_z), (p, q, H*d_z)

        # Use the components to compose the final latent representation
        latent = None
        if compute_task_latent and compute_traj_latent:
            if compute_shared_latent:
                latent = self._compose_latent(shared_latent, task_latent, traj_latent)  # (p, q, H*d_z, m)
            elif use_static_shared_latent:
                assert self._static_shared_latent is not None
                latent = self._compose_latent(self._static_shared_latent, task_latent, traj_latent)  # (p, q, H*d_z, m)
        output = [latent, shared_latent, task_latent, traj_latent]

        if return_reconstructed_input:
            latent = latent.reshape(latent.shape[0], latent.shape[1], -1, self._meta_traj_len)  # (p, q, d_z, m*H)
            reconstructed_input = self._unembedding_head.forward(latent.permute(0, 1, 3, 2))  # (p, q, m*H, d)
            reconstructed_input = reconstructed_input.permute(0, 1, 3, 2)  # (p, q, d, m*H)
            output.append(reconstructed_input)
        return output

    def _do_state_norm(self, trajectories):
        # Clone to keep the input unmodified
        trajectories = trajectories.clone()

        mean = utl.state_rms().mean.unsqueeze(-1)
        var = utl.state_rms().var.unsqueeze(-1) + 1e-8
        if isinstance(trajectories, tuple) or isinstance(trajectories, list):
            trajectories[0] = (trajectories[0] - mean) / torch.sqrt(var)
        elif isinstance(trajectories, torch.Tensor):
            trajectories[..., :self._obs_dim, :] = (trajectories[..., :self._obs_dim, :] - mean) / torch.sqrt(var)
        return trajectories

    def _compose_latent(self, shared_latent, task_latent, traj_latent):
        num_tasks = traj_latent.shape[0]  # p
        num_meta_traj = traj_latent.shape[1]  # q

        if self._task_latent_type in ['diagonal', 'orthogonal']:
            shared_latent = shared_latent.unsqueeze(1)  # (H*d_z, 1)
            shared_latent = shared_latent.repeat(
                num_tasks, num_meta_traj, 1, self._traj_per_meta_traj)  # (p, q, H*d_z, m)
            return shared_latent + self._compute_semi_context(task_latent, traj_latent)  # (p, q, H*d_z, m)
        else:  # self._task_latent_type == 'attention':
            shared_latent = self._shared_embedding(shared_latent)  # (d_e)
            shared_latent = shared_latent.repeat(1, num_tasks*num_meta_traj, 1)  # (1, batch, d_e)

            semi_context = self._compute_semi_context(task_latent, traj_latent)  # (m*H, batch, d_e)

            context = self._context_attention(query=semi_context,
                                              key=shared_latent,
                                              value=shared_latent)  # (m*H, batch, d_e)
            context = context.permute(1, 2, 0)  # (batch, d_e, m*H)
            context = context.reshape(num_tasks, num_meta_traj, -1, self._meta_traj_len)  # (p, q, d_e, m*H)
            return step_to_traj(context, self._traj_len)  # (p, q, d_e*H, m)

    def _compute_semi_context(self, task_latent, traj_latent):
        if self._task_latent_type in ['diagonal', 'orthogonal']:
            if self._task_latent_type == 'diagonal':
                task_latent = torch.diag_embed(task_latent)  # (p, q, H*d_z, H*d_z)
            return torch.matmul(task_latent, traj_latent)  # (p, q, H*d_z, m)
        else:  # self._task_latent_type == 'attention':
            # task_latent   (p, q, d_z)
            # traj_latent   (p, q, H*d_z, m)
            # query=traj_latent, key=value=task_latent
            num_tasks = traj_latent.shape[0]  # p
            num_meta_traj = traj_latent.shape[1]  # q
            batch_size = num_tasks * num_meta_traj

            if not self._task_latent_multidim:
                task_latent = task_latent.reshape(batch_size, -1).unsqueeze(0)  # (1, batch, d_z)
            else:
                task_latent = task_latent.reshape(batch_size, self._meta_traj_len, -1)  # (batch, m*H, d_z)
                task_latent = task_latent.permute(1, 0, 2)  # (m*H, batch, d_z)
                if not self._use_traj_latent_for_context:
                    return task_latent

            traj_latent = traj_to_step(traj_latent, self._traj_len)  # (p, q, d_z, m*H)
            traj_latent = traj_latent.permute(0, 1, 3, 2)  # (p, q, m*H, d_z)
            traj_latent = traj_latent.reshape(batch_size, self._meta_traj_len, -1)  # (batch, m*H, d_z)
            traj_latent = traj_latent.permute(1, 0, 2)  # (m*H, batch, d_z)

            return self._task_attention(query=traj_latent, key=task_latent, value=task_latent)  # (m*H, batch, d_e)

    def compute_linear_coeff(self, task_latent, partial_traj_latent, linear_coeff_head=None):
        """

        :param task_latent: (p', q, H*d_z)
        :param partial_traj_latent: (p', H*d_z, m)
        :param linear_coeff_head:
        :return:
        """
        num_tasks = partial_traj_latent.shape[0]
        linear_coeff_head = self._linear_coeff_head if linear_coeff_head is None else linear_coeff_head

        task_latent = task_latent.flatten(start_dim=1, end_dim=-1)  # (p', q*H*d_z)
        partial_traj_latent = partial_traj_latent.flatten(start_dim=1, end_dim=-1)  # (p', d_z*m*H)

        linear_coeff = linear_coeff_head([task_latent, partial_traj_latent])  # (p', q*m*m)

        if self.args.task_reconstruction_type == 'linear':
            return linear_coeff.reshape(num_tasks, -1, self._traj_per_meta_traj, self._traj_per_meta_traj)  # (p', q, m, m)
        elif self.args.task_reconstruction_type == 'non-linear':
            w0, b0, w1, b1 = linear_coeff  # (p', q*m*w), (p', q*m), (p', q*w*m), (p', q*m)
            w0 = w0.reshape(num_tasks, -1, self._traj_per_meta_traj, self._w_size)  # (p', q, m, w)
            b0 = b0.reshape(num_tasks, -1, 1, self._w_size)  # (p', q, 1, w)
            w1 = w1.reshape(num_tasks, -1, self._w_size, self._traj_per_meta_traj)  # (p', q, w, m)
            b1 = b1.reshape(num_tasks, -1, 1, self._traj_per_meta_traj)  # (p', q, 1, m)
            return w0, b0, w1, b1
        else:
            raise ValueError(f'Expected task_reconstruction_type to be: linear or non-linear. Got {self.args.task_reconstruction_type}')

    def compute_loss(self,
                     data, reconstructed_data,
                     shared_latent, task_latent, traj_latent,
                     partial_task_data, linear_coeff, expected_task_data,
                     use_contrastive_loss, exploration_traj_latent, latent_exploration_target,
                     mask,
                     update=False, pretrain_index=None,
                     print_info=True):
        """

        :param data: (p, q, d, m*H)
        :param reconstructed_data: (p, q, d, m*H)
        :param shared_latent: (H*d_z)
        :param task_latent: (p, q, H*d_z)
        :param traj_latent: (p, q, H*d_z, m)
        :param partial_task_data: (p', 1, d, m*H)
        :param linear_coeff: (p', q, m, m)
        :param expected_task_data: (p', q, d, m*H)
        :param use_contrastive_loss:
        :param exploration_traj_latent: (p', H*d_z, m)
        :param latent_exploration_target: (p')
        :param mask:
        :param update:
        :param pretrain_index:
        :param print_info
        :return:
        """

        loss, log_stats = self._loss_computer.compute_loss(
            data, reconstructed_data,
            shared_latent, task_latent, traj_latent,
            partial_task_data, linear_coeff, expected_task_data,
            use_contrastive_loss, exploration_traj_latent, latent_exploration_target,
            mask,
            print_info,
        )

        if update:
            self.optimiser_transformer.zero_grad()
            loss.backward()
            # TODO Test with clipped gradients
            # clip gradients
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
            # update
            self.optimiser_transformer.step()

        return log_stats

    def finetune_update(self, rl_loss):
        pass
        # TODO Consider combining the RL loss with a reconstruction loss of the task trajectories:
        #  loss = rl_weight * rl_loss + rec_weight * self._loss_computer.compute_loss(...)

        # # self.optimiser_transformer.zero_grad()
        # # rl_loss.backward()
        # # TODO Test with clipped gradients
        # # clip gradients
        # if self.args.encoder_max_grad_norm is not None:
        #     nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
        # # update
        # self.optimiser_transformer.step()
        #
        # # return loss_stats

    def _compute_memory_index(self, trajectories):
        num_tasks = trajectories.shape[0]
        meta_traj_per_task = trajectories.shape[1]

        padding_meta_traj_template = torch.zeros((self._step_size, self._meta_traj_len)).to(global_device())  # (d, m*H)

        # Find the first step that is fully 0 (and assume that is padding)
        mask = torch.all(torch.eq(trajectories, padding_meta_traj_template), dim=-2)  # (p, q, m*H)
        mask = mask.float().masked_fill(mask == 1, float(0.0)).masked_fill(mask == 0, float(1.0))  # (p, q, m*H)

        # FIXME Use method in transformer.py
        # Rearrange the mask into a sequence of trajectories instead of a sequence of steps
        traj_mask = mask.reshape(num_tasks, meta_traj_per_task, self._traj_per_meta_traj, -1)  # (p, q, m, H)
        traj_mask = traj_mask.permute(0, 1, 3, 2)  # (p, q, H, m)
        traj_mask = traj_mask.sum(dim=-2)  # (p, q, m)

        # The index of the last trajectory that is not full padding
        latest_traj_index = F.relu(torch.argmin(traj_mask, dim=-1) - 1)  # (p, q)
        # For any meta-trajectory that has no padding trajectory, return the latest index
        latest_traj_index = torch.where(traj_mask[:, :, -1] > 0,
                                        self._traj_per_meta_traj - 1,
                                        latest_traj_index)  # (p, q)

        # The last non-padding step (NOT meta-step) in each meta-trajectory
        latest_step = torch.gather(traj_mask, dim=-1, index=latest_traj_index.unsqueeze(-1)).squeeze(-1)  # (p, q)
        latest_step = F.relu(latest_step - 1)

        # The last non-padding meta-step of each meta-trajectory
        latest_meta_step = latest_traj_index * self._traj_len + latest_step  # traj_idx * H + step
        return latest_meta_step.type(torch.int64)  # (p, q)

    def _create_prior_trajectory(self, num_tasks, starting_state):
        # We start out with a meta-trajectory of just padding
        trajectories = torch.zeros((num_tasks, 1, self._step_size, self._meta_traj_len),
                                   requires_grad=True).to(global_device())  # (p, 1, d, m*H)

        if starting_state is not None:
            # Add action and reward padding
            act_rew_padding = torch.zeros((num_tasks, self._step_size - starting_state.shape[-1]),
                                          requires_grad=True).to(global_device())  # (p, d_act+1)
            starting_state = torch.cat((starting_state, act_rew_padding), dim=-1)  # (p, d)
            starting_state = starting_state.clone().unsqueeze(1)  # (p, 1, d)
            trajectories[:, :, :, 0] = starting_state
        return trajectories

    def _transform_static_shared_latent(self, value):
        if self.args.static_shared_latent_transformation_type is None:
            return value
        if self.args.static_shared_latent_transformation_type == 'mean':
            return value.mean(dim=1)  # (H*d_z)
        if self.args.static_shared_latent_transformation_type == 'single':
            return value[:, 0]  # (H*d_z)

    def set_static_shared_latent(self, value=None, random=False):
        # Choose only one: give a value or initialise to random
        assert random != bool(value is not None)
        if value is not None:
            if self.args.save_full_shared_latent:
                value = value.mean(dim=-1)  # (d_z, num_batches)
            self._static_shared_latent = self._transform_static_shared_latent(value.clone())  # (d_z)
        if random:
            self._static_shared_latent = torch.rand(self._shared_in)  # (d_z)
        self._static_shared_latent = self._static_shared_latent.to(global_device())

    def compute_static_shared_latent(self, batch_size=None, num_batches=None, update=True, return_full_latent=False):
        with torch.no_grad():
            batch_size = self.args.batch_size if batch_size is None else batch_size
            num_batches = self.args.static_shared_latent_batches if num_batches is None else num_batches

            static_shared_latent = []
            for _ in range(num_batches):
                tasks_batch = self.dataset_storage.get_batch(batch_size=batch_size)  # (p, q, d, m*H)
                _, shared_latent, _, _ = \
                    self.forward(tasks_batch, compute_task_latent=False, compute_traj_latent=False)  # (H*d_z)
                static_shared_latent.append(shared_latent.unsqueeze(dim=-1))

            # If we compute the shared latent vectors from multiple batches, average over all
            if num_batches > 1:
                static_shared_latent = torch.cat(static_shared_latent, dim=-1)  # (H*d_z, num_batches)
                if return_full_latent:
                    full_shared_latent = static_shared_latent.clone()
                static_shared_latent = static_shared_latent.mean(dim=-1)  # (H*d_z)
            else:
                static_shared_latent = static_shared_latent[0]  # (H*d_z, 1)
                if return_full_latent:
                    full_shared_latent = static_shared_latent.clone()
                static_shared_latent = static_shared_latent.squeeze(dim=-1)  # (H*d_z)

            if update:
                self.set_static_shared_latent(static_shared_latent)

            if return_full_latent:
                return full_shared_latent
            return static_shared_latent

    @property
    def optimiser_transformer(self):
        return self._optimiser

    @property
    def unembedding_head(self):
        return self._unembedding_head

    @property
    def linear_coeff_head(self):
        return self._linear_coeff_head
