from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.modules.linear import _LinearWithBias


class SoftHardMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads,
                 dropout_p=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, query: torch.Tensor, key: torch.Tensor, hard_attention=None) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_len, bsz, embed_dim = query.size()
        head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5

        _b = self.in_proj_bias
        _start = 0
        _end = embed_dim
        _w = self.in_proj_weight[_start:_end, :]
        _b = _b[_start:_end]
        q = F.linear(query, _w, _b)
        _b = self.in_proj_bias
        _start = embed_dim
        _end = None
        _w = self.in_proj_weight[_start:, :]
        _b = _b[_start:]
        k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        q = q * scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout_p, training=self.training)
        if hard_attention is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            hard_attention = hard_attention.unsqueeze(1)
            if hard_attention.dim() == 3:
                hard_attention = hard_attention.unsqueeze(1)
            hard_attention = hard_attention.repeat(1, self.num_heads, 1, 1)
            attn_output_weights = attn_output_weights * hard_attention
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
        return attn_output, attn_output_weights
