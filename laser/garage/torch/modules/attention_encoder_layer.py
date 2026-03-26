"""A more general implementation of laser.garage.torch.modules.TransformerEncoderLayerNoLN where the query, key, and value
used in multi-head attention can be different"""

from typing import Optional

import torch.nn.functional as F
from torch import Tensor
from torch import nn

from laser.garage.torch.modules.transformer_no_ln import _get_activation_fn


class AttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", use_residual=True):
        super(AttentionEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.use_residual = use_residual

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(AttentionEncoderLayer, self).__setstate__(state)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(query=query, key=key, value=value,
                              attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False)[0]
        src = query + self.dropout1(src2) if self.use_residual else src2
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2) if self.use_residual else src2
        return src
