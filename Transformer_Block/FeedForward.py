import torch
from torch import nn
from torch import Tensor
import math

class FeedForward(nn.Module):
    def __init__(self,
        hidden_size:int,
        expand_size:int,
        act:nn.Module=nn.GELU,
        drop:float=0.1,
        bias:bool=True,
    ):
        super().__init__()
        # project input to expanded dimension
        self.fc1 = nn.Linear(hidden_size, expand_size, bias=bias)
        # activation function to introduce non-linearity
        self.act = act()
        # project back to the input dimension
        self.fc2 = nn.Linear(expand_size, hidden_size, bias=bias)
        # optional dropout layer to prevent overfitting
        self.drop = nn.Dropout(drop)

    def forward(self, x:Tensor):
        x = self.fc1(x) # apply first linear layer
        x = self.act(x) # apply activation function
        x = self.fc2(x) # apply second linear layer
        x = self.drop(x) # optionally apply dropout layer
        return x