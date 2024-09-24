from torch import nn
import math
from torch import Tensor

class SingleHeadSelfAttention(nn.Module):
    '''
    we will start merge Wq, Wk, Wv into single linear layer, Wqkv, and unbind the outputs into Q, K, and V.
    This is accomplished by increasing the output shape by a factor of three . This is mathematically equivalent to three individual linear layers, each with the same input and output shape.
    Alternatively, we might use two layers, one for Q and one for both K and V for implementing KV caching.
    '''
    def __init__(self,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.Wqkv = nn.Linear(hidden_size, (hidden_size//4)*3, bias=bias)
        self.Wo = nn.Linear(hidden_size//4, hidden_size, bias=bias)

    def forward(self, x:Tensor):
        B, S, C = x.shape
        q, k, v = self.Wqkv(x).reshape(B, S, 3, C//4).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))
        attn = attn.softmax(dim=-1)
        x = attn @ v
        return self.Wo(x)