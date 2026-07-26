from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import TransformerBlock


class MultiScaleOverlapTokenizer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, branches: Sequence[Dict[str, Sequence[int]]]):
        super().__init__()
        if len(branches) not in {1, 2}:
            raise ValueError("tokenizer requires one or two branches")
        self.branch_specs = list(branches)
        self.projections = nn.ModuleList()
        for spec in branches:
            patch = tuple(int(value) for value in spec["patch_size"])
            stride = tuple(int(value) for value in spec["stride"])
            padding = tuple(value // 2 for value in patch)
            self.projections.append(nn.Conv2d(in_channels, embed_dim, patch, stride, padding))
        self.fuse = nn.Conv2d(embed_dim * len(branches), embed_dim, 1) if len(branches) > 1 else nn.Identity()

    @property
    def branch_count(self) -> int:
        return len(self.projections)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        maps = [projection(images) for projection in self.projections]
        anchor_size = maps[0].shape[-2:]
        maps = [maps[0]] + [F.interpolate(item, anchor_size, mode="bilinear", align_corners=False) for item in maps[1:]]
        feature_map = self.fuse(torch.cat(maps, dim=1)) if len(maps) > 1 else maps[0]
        tokens = feature_map.flatten(2).transpose(1, 2)
        return tokens, (int(anchor_size[0]), int(anchor_size[1]))


class OverlappingVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: Sequence[int],
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        branches: Sequence[Dict[str, Sequence[int]]],
    ):
        super().__init__()
        self.image_size = tuple(int(value) for value in image_size)
        self.tokenizer = MultiScaleOverlapTokenizer(3, embed_dim, branches)
        patch = branches[0]["patch_size"]
        stride = branches[0]["stride"]
        base_h = math.floor((self.image_size[0] + 2 * (patch[0] // 2) - patch[0]) / stride[0] + 1)
        base_w = math.floor((self.image_size[1] + 2 * (patch[1] // 2) - patch[1]) / stride[1] + 1)
        self.base_grid = (base_h, base_w)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, 1 + base_h * base_w, embed_dim))
        rates = torch.linspace(0, drop_path, depth).tolist() if depth else []
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, rates[index]) for index in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def _position_for(self, grid: Tuple[int, int]) -> torch.Tensor:
        if grid == self.base_grid:
            return self.position
        class_position = self.position[:, :1]
        spatial = self.position[:, 1:].reshape(1, self.base_grid[0], self.base_grid[1], -1).permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial, grid, mode="bicubic", align_corners=False)
        spatial = spatial.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], -1)
        return torch.cat((class_position, spatial), dim=1)

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        spatial, grid = self.tokenizer(images)
        class_token = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_token, spatial), dim=1) + self._position_for(grid)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(images)[:, 0]

