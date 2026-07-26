from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return tensor
        keep = 1.0 - self.probability
        shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
        mask = tensor.new_empty(shape).bernoulli_(keep)
        return tensor * mask / keep


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        normalized = self.norm1(tokens).transpose(0, 1)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.drop_path(attended.transpose(0, 1))
        return tokens + self.drop_path(self.mlp(self.norm2(tokens)))

