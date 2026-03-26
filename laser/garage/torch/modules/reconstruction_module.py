import torch
from laser.garage.torch.modules.mlp_module import MultiHeadedMLPModule


class ReconstructionModule(MultiHeadedMLPModule):
    def __init__(self,
                 input_dim,
                 obs_dim,
                 act_dim,
                 rew_dim,
                 hidden_sizes,
                 hidden_nonlinearity=torch.relu,
                 layer_normalization=False
                 ):
        # A multiheaded network that returns 3 outputs: reconstructed obs, act and rew
        super(ReconstructionModule, self).__init__(
            n_heads=3,
            input_dim=input_dim,
            output_dims=(obs_dim, act_dim, rew_dim),
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearities=None,
            layer_normalization=layer_normalization,
        )

    def forward(self, transformer_embedded_output):
        obs_out, act_out, rew_out = super().forward(transformer_embedded_output)  # (p, q, m*H, d_obs/d_act/1)

        # Return the reconstructed meta-trajectory
        return torch.cat((obs_out, act_out, rew_out), dim=-1)  # (p, q, m*H, d)
