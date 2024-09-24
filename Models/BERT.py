import torch
from torch import nn
from Self_Attention import SelfAttention, CausalAttention, CausalCrossAttention, MultiHeadAttention, BidirectionalAttention
from Transformer_Block import TransformerBlock, PositionalEncoding
from torch import Tensor, BoolTensor
import math

class BERT(nn.Module):
    def __init__(self, num_layers:int, vocab_size:int, hidden_size:int, num_heads:int,
                 context_size:int, expand_size:int, attention:nn.Module=BidirectionalAttention,
                 act:nn.Module=nn.GELU, embed_drop:float=0.1, attn_drop:float=0.1,
                 out_drop:float=0.1, ffn_drop:float=0.1, head_norm:bool=True,
                 tie_weights:bool=True, head_bias:bool=True, bias:bool=True):
        super().__init__()
        # initialize vocab & positional encodings to convert numericalied tokens
        # & position indicies to token and position vectors, with optional dropout
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_encode = PositionalEncoding(context_size, hidden_size)
        self.embed_drop = nn.Dropout(embed_drop)

        # initialize num_layers of transformer layers
        self.tfm_blocks = nn.ModuleList([TransformerBlock(
                hidden_size=hidden_size, num_heads=num_heads, expand_size=expand_size,
                attention=attention, act=act, bias=bias, attn_drop=attn_drop,
                out_drop=out_drop, ffn_drop=ffn_drop)
            for _ in range(num_layers)])

        # optional pre-head normalization
        self.head_norm = nn.LayerNorm(hidden_size) if head_norm else nn.Identity()

        # predicts the next token in the sequence
        self.head = nn.Linear(hidden_size, vocab_size, bias=head_bias)

        # optionally set the vocab embedding and prediction head to share weights
        if tie_weights:
            self.head.weight = self.vocab_embed.weight

        self.apply(self._init_weights)

    def forward(self, x: Tensor, return_preds:bool=True):
        # convert numericalized tokens of shape (B, S)
        # into token embeddings of shape (B, S, C)
        tokens = self.vocab_embed(x)
        # positional encodings are shape (S, C)
        pos = self.pos_encode(x)

        # positional encodings are added to token embeddings
        x = self.embed_drop(tokens + pos)

        # pass token vectors through all transformer layers
        for block in self.tfm_blocks:
            x = block(x)

        # apply optional pre-head normalization
        x = self.head_norm(x)

        # if MLM pretraining, don't predict outputs here
        if return_preds:
            # converts input token vectors of shape (B, S, C) to probability
            # distribution of shape batch, sequence length, vocabulary size (B, S, VS)
            return self.head(x)
        else:
            return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class BERTForMaskedLM(BERT):
    def __init__(self, loss_fn:nn.Module=nn.CrossEntropyLoss(),
                 mlm_prob:float|None=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.mlm_prob = mlm_prob

    def forward(self, input_ids: Tensor, labels: Tensor, mlm_prob: float|None = None):
        x = super().forward(input_ids, False)

        # flatten both the labels and the intermediate outputs
        labels = labels.view(-1)
        x = x.view(labels.shape[0], -1)

        # only select the masked tokens for predictions
        mask_tokens = labels != self.loss_fn.ignore_index

        # torch.compile with fullgraph cannot have dynamic shapes
        # if `mlm_prob` is set, this will create workable indicies
        # if `mlm_prob` is None, then fullgraph=True cannot be used
        mlm_prob = self.mlm_prob if mlm_prob is None else mlm_prob
        if mlm_prob is not None:
            num_masks = math.floor(self.mlm_prob * mask_tokens.shape[0])
        else:
            num_masks = mask_tokens.sum().int()
        indices = torch.argsort(mask_tokens.int())[-num_masks:]

        # selecting the masked tokens reshapes x to (B*S, VS) and labels to (B*S)
        x = x[indices]
        labels = labels[indices]

        # converts input token vectors of shape (B*S, C)
        # to probability distribution of shape (B*S, VS)
        logits = self.head(x)

        # return both the logits and the loss
        return {'logits': logits, 'loss': self.loss_fn(logits, labels)}