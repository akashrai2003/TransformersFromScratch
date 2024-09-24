import torch
from torch import nn
from Self_Attention import SelfAttention, CausalAttention, CausalCrossAttention, MultiHeadAttention, BidirectionalAttention
from Transformer_Block import TransformerBlock
from torch import Tensor, BoolTensor
import math

class GPT(nn.Module):
    def __init__(self, num_layers:int, vocab_size:int, hidden_size:int, num_heads:int,
                 context_size:int, expand_size:int, attention:nn.Module=CausalAttention,
                 act:nn.Module=nn.GELU, embed_drop:float=0.1, attn_drop:float=0.1,
                 out_drop:float=0.1, ffn_drop:float=0.1, head_norm:bool=True,
                 tie_weights:bool=True, head_bias:bool=True, bias:bool=True):
        super().__init__()
        # initialize vocab & positional embeddings to convert numericalied tokens
        # & position indicies to token and position vectors, with optional dropout
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(context_size, hidden_size)
        self.embed_drop = nn.Dropout(embed_drop)

        # initialize num_layers of transformer layers
        self.tfm_blocks = nn.ModuleList([TransformerBlock(
                hidden_size=hidden_size, num_heads=num_heads, context_size=context_size,
                expand_size=expand_size, attention=attention, act=act, bias=bias,
                attn_drop=attn_drop, out_drop=out_drop, ffn_drop=ffn_drop)
            for _ in range(num_layers)])

        # optional pre-head normalization
        self.head_norm = nn.LayerNorm(hidden_size) if head_norm else nn.Identity()

        # predicts the next token in the sequence
        self.head = nn.Linear(hidden_size, vocab_size, bias=head_bias)

        # optionally set the vocab embedding and prediction head to share weights
        if tie_weights:
            self.head.weight = self.vocab_embed.weight

        # precreate positional indices for the positional embedding
        pos = torch.arange(0, context_size, dtype=torch.long)
        self.register_buffer('pos', pos, persistent=False)

        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        # convert numericalized tokens of shape (B, S)
        # into token embeddings of shape (B, S, C)
        tokens = self.vocab_embed(x)
        # positional embeddings are shape (S, C)
        pos = self.pos_embed(self.pos[:x.shape[1]])

        # positional embeddings are added to token embeddings
        x = self.embed_drop(tokens + pos)

        # pass token vectors through all transformer layers
        for block in self.tfm_blocks:
            x = block(x)

        # apply optional pre-head normalization
        x = self.head_norm(x)

        # converts input token vectors of shape (B, S, C) to probability
        # distribution of shape batch, sequence length, vocabulary size (B, S, VS)
        return self.head(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == 'fc2':
                # GPT-2 style FFN init
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class GPTForCausalLM(GPT):
    def __init__(self, loss_fn:nn.Module=nn.CrossEntropyLoss(), **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def forward(self, x: Tensor):
        # the labels are the next token, so shift the labels over one
        # & resize inputs to same length as labels by dropping last token
        inputs = x[:, :-1]
        labels = x[:, 1:]

        # logits are of shape batch, sequence length, vocab size (B, S, VS),
        # labels are of shape batch, vocab size (B, S)
        logits = super().forward(inputs)

        # flatten logits into (B*S, VS) and labels into (B*S) & calculate loss
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        # return both the logits and the loss
        return {'logits': logits, 'loss': loss}