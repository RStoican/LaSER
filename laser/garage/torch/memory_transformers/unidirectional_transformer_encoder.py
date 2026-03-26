from abc import ABC

import torch
from laser.garage.torch._functions import global_device
from laser.garage.torch.memory_transformers.transformer_encoder import TransformerEncoder
from laser.garage.torch.modules import MLPModule

from laser.garage.utils.helpers import step_to_traj


class UnidirectionalTransformerEncoder(TransformerEncoder, ABC):
    def _create_latent_heads(self):
        if self._multiheaded_latent_network:
            raise NotImplementedError

        self._traj_latent_head = MLPModule(input_dim=self._d_model,  # d_model
                                           output_dim=self._out_latent_dim,  # d_z
                                           hidden_sizes=self._latent_hidden_sizes,
                                           hidden_nonlinearity=self._latent_hidden_nonlinearity,
                                           hidden_w_init=self._latent_hidden_w_init,
                                           hidden_b_init=self._latent_hidden_b_init,
                                           output_nonlinearity=self._latent_output_nonlinearity,
                                           output_w_init=self.mlp_output_w_init,
                                           output_b_init=self._latent_output_b_init,
                                           layer_normalization=self._latent_layer_normalization, ) #if self._task_latent_type != 'attention' else None

    # Get the output from a unidirectional encoder, and compute the trajectory latent
    def _compute_latent(self, transformer_output, compute_shared_latent=True, compute_task_latent=True):
        # traj_latent = transformer_output
        # if self._task_latent_type != 'attention':
        #     traj_latent = self._traj_latent_head(traj_latent)  # (p, q, m*H, d_z)
        traj_latent = self._traj_latent_head(transformer_output)  # (p, q, m*H, d_z)

        # Convert the sequence of m*h d-dimensional steps to a sequence of m H*d-dimensional trajectories
        traj_latent = traj_latent.permute(0, 1, 3, 2)  # (p, q, d_z, m*H)
        return step_to_traj(traj_latent, self._traj_len)  # (p, q, H*d_z, m)

    # Unidirectional attention masks all future tokens
    def _get_attention_mask(self, seq_len=None):
        if self.src_mask is not None:
            return self.src_mask

        sz = self._meta_traj_len if seq_len is None else seq_len  # m*H
        ones = torch.ones(sz, sz).to(global_device())  # (m*H, m*H)
        # Create a matrix where attention between a step and its future steps is zero,
        # i.e. only bottom-left triangle is 1, the rest is 0
        mask = (torch.triu(ones) == 1).transpose(0, 1)  # (m*H, m*H)
        # Add -inf to the attention of all future steps. Add nothing to the attention of past steps
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  # (m*H, m*H)
        self.src_mask = mask
        return mask

    def _is_valid_encoder(self):
        return self._shared_latent_head is None \
            and self._task_latent_head is None \
            and self._traj_latent_head is not None \
            and self._shared_latent_pooling is None
