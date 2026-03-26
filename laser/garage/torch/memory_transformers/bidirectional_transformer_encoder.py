from abc import ABC

from laser.garage.torch.memory_transformers.transformer_encoder import TransformerEncoder
from laser.garage.torch.modules import MLPModule
from torch import nn


class BidirectionalTransformerEncoder(TransformerEncoder, ABC):
    def _create_latent_heads(self):
        if self._multiheaded_latent_network:
            raise NotImplementedError
            return

        # The input dimension of the task latent head
        task_in_dim = self._traj_per_meta_traj * self._d_model  # m*d_model
        if self._task_latent_type == 'orthogonal':
            task_in_dim = self._meta_train_meta_trajs * self._traj_per_meta_traj  # q*m
        if self._task_latent_type == 'attention':
            if self._task_latent_multidim:
                task_in_dim = self._d_model  # d_model
            else:
                task_in_dim = self._meta_traj_len * self._d_model  # m*H*d_model

        # The output dimension of the task latent head
        task_latent_dim = self._out_latent_dim  # (d_z)
        if self._task_latent_type == 'orthogonal':
            task_latent_dim *= self._traj_len  # (H*d_z)

        self._task_latent_head = MLPModule(input_dim=task_in_dim,
                                           output_dim=task_latent_dim,
                                           hidden_sizes=self._latent_hidden_sizes,
                                           hidden_nonlinearity=self._latent_hidden_nonlinearity,
                                           hidden_w_init=self._latent_hidden_w_init,
                                           hidden_b_init=self._latent_hidden_b_init,
                                           output_nonlinearity=self._latent_output_nonlinearity,
                                           output_w_init=self.mlp_output_w_init,
                                           output_b_init=self._latent_output_b_init,
                                           layer_normalization=self._latent_layer_normalization, )

        # We only need a shared latent head during training
        if self._train_mode:
            # A pooling layer for the shared latent. The output of the transformer is too large for the shared MLP to
            # be feasible. So reduce the transformer output through pooling
            # For each task, we will reduce its dimension to shared_pool_dimension, then combine all tasks into one
            # vector and compute the shared latent
            if self._shared_latent_pooling_type is None:
                self._shared_latent_pooling = None
                shared_in_dim \
                    = self._meta_train_tasks * self._meta_train_meta_trajs \
                    * self._d_model * self._meta_traj_len  # p*q*d_model*m*H
            else:
                # Using only p and q to compute the pooled output should help make the output size more
                # environment agnostic, e.g. not dependent on the horizon H or the m-shot adaptation
                if not self._simple_shared_latent:
                    shared_latent_pooled_out = self._meta_train_tasks * self._meta_train_meta_trajs  # p*q
                else:
                    shared_latent_pooled_out = self._traj_per_meta_traj  # m
                if self._shared_latent_pooling_type == 'max':
                    self._shared_latent_pooling = nn.AdaptiveMaxPool1d(shared_latent_pooled_out)
                elif self._shared_latent_pooling_type == 'avg':
                    self._shared_latent_pooling = nn.AdaptiveAvgPool1d(shared_latent_pooled_out)
                else:
                    ValueError(f'Expected shared_latent_pooling_type to be max, avg, or none. '
                               f'Got {self._shared_latent_pooling_type}')
                if not self._simple_shared_latent:
                    shared_in_dim = shared_latent_pooled_out * self._d_model  # p*q*d_model
                else:
                    shared_in_dim = shared_latent_pooled_out * self._meta_train_tasks * self._meta_train_meta_trajs  # p*q*m

            self._shared_latent_head = MLPModule(input_dim=shared_in_dim,
                                                 output_dim=self._out_latent_dim,  # d_z
                                                 hidden_sizes=self._latent_hidden_sizes,
                                                 hidden_nonlinearity=self._latent_hidden_nonlinearity,
                                                 hidden_w_init=self._latent_hidden_w_init,
                                                 hidden_b_init=self._latent_hidden_b_init,
                                                 output_nonlinearity=self._latent_output_nonlinearity,
                                                 output_w_init=self.mlp_output_w_init,
                                                 output_b_init=self._latent_output_b_init,
                                                 layer_normalization=self._latent_layer_normalization, )

    # Get the output from a bidirectional encoder, and compute the shared latent and the task latent
    def _compute_latent(self, transformer_output, compute_shared_latent=True, compute_task_latent=True):
        num_tasks = transformer_output.shape[0]
        num_meta_traj = transformer_output.shape[1]

        shared_latent, task_latent = None, None
        if compute_shared_latent:
            if self._shared_latent_pooling is not None:
                # Pool over all transformer output features, independently
                if self._simple_shared_latent:
                    # shared_input = shared_input.permute(1, 0, 2)  # (H, p*q*m, d_model)
                    # shared_input = shared_input.reshape(self._traj_len, -1)  # (H, p*q*m*d_model)
                    # shared_input = self._shared_latent_pooling(shared_input)  # (H, p*q)
                    # shared_input = shared_input.reshape(-1)  # (H*p*q)

                    shared_input = transformer_output.view(
                        self._meta_train_tasks, self._meta_train_meta_trajs, -1)  # (p, q, m*H*d_model)
                    shared_input = self._shared_latent_pooling(shared_input)  # (p, q, m)
                    shared_input = shared_input.reshape(-1)  # (p*q*m)

                    shared_latent = self._shared_latent_head(shared_input)  # (d_z)
                else:
                    shared_input = transformer_output.view(-1, self._traj_len, self._d_model)  # (p*q*m, H, d_model)
                    shared_input = shared_input.permute(1, 2, 0)  # (H, d_model, p*q*m)
                    shared_input = self._shared_latent_pooling(shared_input)  # (H, d_model, p*q)
                    # Combine each step's features into a single vector
                    shared_input = shared_input.view(self._traj_len, -1)  # (H, p*q*d_model)

                    shared_latent = self._shared_latent_head(shared_input)  # (H, d_z)
                    shared_latent = shared_latent.reshape(-1)  # (H*d_z)
            else:
                raise NotImplementedError
                shared_input = transformer_output.view(-1)  # (p*q*m*H*d_model)

        if compute_task_latent:
            if self._task_latent_type == 'attention':
                if self._task_latent_multidim:
                    task_latent = self._task_latent_head(transformer_output)  # (p, q, m*H, d_z)
                else:
                    task_input = transformer_output.reshape(num_tasks, num_meta_traj, -1)  # (p, q, m*H*d_model)
                    # FIXME Should we permute(0, 1, 3, 2, 4) ???
                    task_latent = self._task_latent_head(task_input)  # (p, q, d_z)

            elif self._task_latent_type == 'diagonal':
                task_input = transformer_output.view(
                    num_tasks, num_meta_traj, self._traj_per_meta_traj, self._traj_len, -1)  # (p, q, m, H, d_model)
                task_input = task_input.permute(0, 1, 3, 2, 4)  # (p, q, H, m, d_model)
                task_input = task_input.reshape(num_tasks, num_meta_traj, self._traj_len, -1)  # (p, q, H, m*d_model)

                task_latent = self._task_latent_head(task_input)  # (p, q, H, d_z)
                task_latent = task_latent.reshape(num_tasks, num_meta_traj, -1)  # (p, q, H*d_z)

            elif self._task_latent_type == 'orthogonal':
                raise NotImplementedError('Not tested')
                # FIXME Not tested
                task_input = transformer_output.view(
                    num_tasks, num_meta_traj, self._traj_per_meta_traj, self._traj_len, -1)  # (p, q, m, H, d_model)
                task_input = task_input.permute(0, 3, 4, 1, 2)  # (p, H, d_model, q, m)
                task_input = task_input.reshape(
                    num_tasks, -1, self._meta_train_meta_trajs * self._traj_per_meta_traj)  # (p, H*d_model, q*m)
                task_latent = self._task_latent_head(task_input)  # (p, d_model*H, d_z*H)

            else:
                raise ValueError(f'Expected task_latent_type to be: diagonal, orthogonal. Got {self._task_latent_type}')
        return shared_latent, task_latent

    # Bidirectional attention requires no mask: each token attends to all other tokens (both past and future)
    def _get_attention_mask(self, seq_len=None):
        return None

    def _is_valid_encoder(self):
        return (self._shared_latent_head is not None or not self._train_mode) \
            and self._task_latent_head is not None \
            and self._traj_latent_head is None \
            and (self._shared_latent_pooling is not None or not self._train_mode)
