from torch import nn
from . import selfAttention
from torch import Tensor
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,
    hidden_size: int,
    num_heads: int,
    bias: bool = True,
    ):
        # input dimension must be divisible by num_heads
        assert hidden_size % num_heads == 0
        # number of attention heads
        self.nh = num_heads
        super().__init__()
        # linear layer to project queries, keys, values
        self.Wqkv = nn.Linear(hidden_size, hidden_size*3, bias=bias)
        # linear layer to project final output
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: Tensor):
        B, S, C = x.shape

        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        attn = attn.softmax(dim=-1)

        x = attn @ v

        return self.Wo(x.transpose(1, 2).reshape(B, S, C))