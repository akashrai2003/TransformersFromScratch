import torch
from torch import nn
from Self_Attention import SelfAttention, CausalAttention, CausalCrossAttention, MultiHeadAttention, BidirectionalAttention
from torch import Tensor, BoolTensor
import math

class PositionalEncoding(nn.Module):
    def __init__(self,
        context_size: int,
        hidden_size: int
    ):
        super().__init__()
        # create the positional encoding tensor of shape
        # maximum sequence length (MS) by embedding dimension (C)
        pe = torch.zeros(context_size, hidden_size, dtype=torch.float)

        # pre-populate the position and the div_terms
        position = torch.arange(context_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000) / hidden_size)
        )

        # even positional encodings use sine, odd cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as a buffer so autograd doesn't modify
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor):
        # return the pre-calculated positional encodings
        # up to sequence length (S). output shape (1, S, C)
        return self.pe[:, :x.shape[1], :]