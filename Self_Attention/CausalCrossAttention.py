import torch
from torch import nn
from . import selfAttention
from torch import Tensor, BoolTensor
import math

class CausalCrossAttention(nn.Module):
    def __init__(self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        attn_drop: float = 0.1,
        out_drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.nh = num_heads
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.Wkv = nn.Linear(hidden_size, hidden_size * 2, bias=bias)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)
        self.register_buffer('causal_mask',
            torch.triu(torch.ones([context_size, context_size],
                       dtype=torch.bool), diagonal=1)
                .view(1, 1, context_size, context_size))

    def forward(self, x: Tensor, y: Tensor, mask: BoolTensor):
        B, S, C = x.shape

        q = self.Wq(x).reshape(B, S, self.nh, C//self.nh).transpose(1, 2)
        y = self.Wkv(y).reshape(B, S, 2, self.nh, C//self.nh)
        k, v = y.transpose(3, 1).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        combined_mask = self.causal_mask + mask.view(B, 1, 1, S)
        attn = attn.masked_fill(combined_mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v

        x = x.transpose(1, 2).reshape(B, S, C)
        return self.out_drop(self.Wo(x))