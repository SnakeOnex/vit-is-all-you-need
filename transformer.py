import torch, torch.nn as nn, torch.nn.functional as F, time
from dataclasses import dataclass
from einops import rearrange

@dataclass
class TransformerConfig:
    n_layers: int
    n_heads: int
    n_embd: int
    block_size: int
    def __post_init__(self):
        self.head_dim = self.n_embd // self.n_heads

class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Attention, self).__init__()
        for k, v in config.__dict__.items(): setattr(self, k, v)
        self.qkv = nn.Linear(self.n_embd, self.n_embd * 3)
        self.register_buffer("attn_mask", ~torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)))
    def forward(self, x):
        q, k, v = rearrange(self.qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.n_heads)
        attn = q @ k.transpose(-2,-1) * (1/self.head_dim)**0.5
        attn.masked_fill_(self.attn_mask, float("-inf"))
        out = attn.softmax(dim=-1) @ v
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
