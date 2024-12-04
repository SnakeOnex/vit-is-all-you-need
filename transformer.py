import torch, torch.nn as nn, torch.nn.functional as F, time
from dataclasses import dataclass
from einops import rearrange

@dataclass
class TransformerConfig:
    n_layers: int
    n_heads: int
    n_embd: int
    block_size: int
    causal: bool = False
    dropout: float = 0.0
    def __post_init__(self):
        self.head_dim = self.n_embd // self.n_heads

class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Attention, self).__init__()
        if "causal" not in config.__dict__: config.causal = False #HACK: for backwards compatibility with old configs
        for k, v in config.__dict__.items(): setattr(self, k, v)
        self.qkv = nn.Linear(self.n_embd, self.n_embd * 3)
        if self.causal:
            mask = torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))  # Convert 1s to -inf
            self.register_buffer("mask", mask)
    def forward(self, x):
        q, k, v = rearrange(self.qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.n_heads)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, attn_mask=self.mask[:q.size(2), :q.size(2)] if self.causal else None)
        return rearrange(out, "b h n d -> b n (h d)", h=self.n_heads)

class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerLayer, self).__init__()
        for k, v in config.__dict__.items(): setattr(self, k, v)
        self.multi_attn = Attention(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.GELU(),
            nn.Linear(4 * self.n_embd, self.n_embd),
            nn.Dropout(self.dropout)
        )
    def forward(self, x):
        x = x + self.multi_attn(F.layer_norm(x, (self.n_embd,)))
        x = x + self.mlp(F.layer_norm(x, (self.n_embd,)))
        return x

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()
        for k, v in config.__dict__.items(): setattr(self, k, v)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

def S(**kwargs): return TransformerConfig(n_layers=6, n_heads=8, n_embd=512, **kwargs)
def B(**kwargs): return TransformerConfig(n_layers=12, n_heads=12, n_embd=768, **kwargs)
def L(**kwargs): return TransformerConfig(n_layers=24, n_heads=16, n_embd=1024, **kwargs)
transformer_configs = {"S": S, "B": B, "L": L}
