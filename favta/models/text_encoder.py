from __future__ import annotations

import torch
import torch.nn as nn

from .layers import TransformerBlock


class TextTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, length: int, embed_dim: int, depth: int, heads: int, dropout: float):
        super().__init__()
        self.length = int(length)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, heads, 4.0, dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > self.length:
            raise ValueError("text token tensor must be [batch, length<=configured_length]")
        padding_mask = token_ids.eq(0)
        tokens = self.embedding(token_ids) + self.position[:, : token_ids.shape[1]]
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=padding_mask)
        tokens = self.norm(tokens)
        lengths = token_ids.ne(0).sum(dim=1).clamp_min(1) - 1
        return tokens[torch.arange(tokens.shape[0], device=tokens.device), lengths]

