import torch
from torch import nn
from . import selfAttention
from torch import Tensor, BoolTensor
import math
class BidirectionalAttention(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int, attn_drop:float=0.1,
                 out_drop:float=0.1, bias:bool=True):
        super().__init__()
        # input dimension must be divisible by num_heads
        assert hidden_size % num_heads == 0
        # number of Attention heads
        self.nh = num_heads

        # linear layer to project queries, keys, values
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)

        # attention dropout layer to prevent overfitting
        self.attn_drop = nn.Dropout(attn_drop)

        # linear layer to project final output
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

        # final output dropout layer to prevent overfitting
        self.out_drop = nn.Dropout(out_drop)

    # boolean `mask` of shape (batch_size, sequence_length)
    # where True is masked and False is unmasked
    def forward(self, x: Tensor, mask: BoolTensor|None = None):
        # batch size, sequence length, input dimension
        B, S, C = x.shape

        # split into queries, keys, & values of shape
        # batch size (B), num_heads (NH), sequence length (S), head size (HS)
        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        # dot product queries and keys for each head
        # (B, NH, S, S) = (B, NH, S, HS) @ (B, NH, HS, S)
        attn = q @ k.transpose(-2, -1)

        # scale by square root of output dimension
        attn = attn / math.sqrt(k.size(-1))

        # reshape and mask attention scores
        if mask is not None:
            attn = attn.masked_fill(mask.view(B, 1, 1, S), float('-inf'))

        # apply softmax to get attention weights
        attn = attn.softmax(dim=-1)

        # apply dropout to attention weight
        attn = self.attn_drop(attn)

        # dot product attention weights with values of shape
        # (B, NH, S, HS) = (B, NH, S, S) @ (B, NH, HS, S)
        x = attn @ v

        # and transpose heads & sequence and reshape back to (B, S, C)
        x = x.transpose(1, 2).reshape(B, S, C)

        # apply final linear layer and dropout to get output (B, S, C)
        return self.out_drop(self.Wo(x))