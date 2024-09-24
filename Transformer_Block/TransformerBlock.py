import torch
from torch import nn
from Self_Attention import SelfAttention, CausalAttention, CausalCrossAttention, MultiHeadAttention, BidirectionalAttention
from torch import Tensor, BoolTensor
import math
from Transformer_Block import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int, expand_size:int,
                 attention:nn.Module=BidirectionalAttention, act:nn.Module=nn.GELU,
                 attn_drop:float=0.1, out_drop:float=0.1, ffn_drop:float=0.1,
                 bias:bool=True):
        super().__init__()
        # first pre-norm layer
        self.norm1 = nn.LayerNorm(hidden_size)
        # initialize the attention layer
        self.attn = attention(
            hidden_size=hidden_size, num_heads=num_heads, attn_drop=attn_drop,
            out_drop=out_drop, bias=bias
        )

        # second pre-norm layer
        self.norm2 = nn.LayerNorm(hidden_size)
        # initialize the feed forward network (MLP)
        self.ffn = FeedForward(
            hidden_size=hidden_size, expand_size=expand_size, act=act,
            drop=ffn_drop, bias=bias,
        )

    def forward(self, x: Tensor):
        # normalize input then add residual to attention output
        x = x + self.attn(self.norm1(x))

        # normalize input then add residual to feedforward output
        return x + self.ffn(self.norm2(x))