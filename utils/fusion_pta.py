import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelTemporalAdapter(nn.Module):
    """
    Parallel Temporal Adapter on [B, T, D].
    """
    def __init__(
        self,
        dim: int = 768,
        adapter_ratio: float = 0.25,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        hidden_dim = int(dim * adapter_ratio)
        hidden_dim = max(hidden_dim, 32)
        padding = kernel_size // 2

        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden_dim)
        self.temporal = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        y = self.down(y)           # [B, T, H]
        y = y.transpose(1, 2)      # [B, H, T]
        y = self.temporal(y)
        y = y.transpose(1, 2)      # [B, T, H]
        y = self.act(y)
        y = self.dropout(y)
        y = self.up(y)             # [B, T, D]
        return residual + y


class PTAFusion(nn.Module):
    """
    OmniCLIP-inspired PTA fusion.

    Input:  x [B, T, D]
    Output: z [B, D]
    """
    def __init__(
        self,
        dim: int = 768,
        num_adapters: int = 5, #2
        adapter_ratio: float = 0.25,
        kernel_size: int = 3,
        dropout: float = 0.1,
        agg: str = "attn"
    ):
        super().__init__()
        assert agg in ["mean", "max", "attn", "mlp"]

        self.adapters = nn.ModuleList([
            ParallelTemporalAdapter(dim, adapter_ratio, kernel_size, dropout)
            for _ in range(num_adapters)
        ])

        self.agg = agg
        if agg == "attn":
            self.attn_score = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1)
            )
        elif agg == "mlp":
            d_model = dim
            self.attn_score = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1)
            )
            self.mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
    def aggregate(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        if self.agg == "mean":
            return x.mean(dim=1)
        elif self.agg == "max":
            return x.max(dim=1).values
        elif self.agg == "attn":
            score = self.attn_score(x)
            weight = torch.softmax(score, dim=1)
            return (x * weight).sum(dim=1)
        elif self.agg == "mlp":

            weight_x = torch.softmax(self.attn_score(x), dim=1)
            x = (x * weight_x).sum(dim=1)

            weight_raw = torch.softmax(self.attn_score(raw), dim=1)
            raw = (raw * weight_raw).sum(dim=1)

            inputs = torch.cat([x, raw], dim=-1)
            return self.mlp(inputs)
        else:
            raise ValueError(self.agg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected [B, T, D], got {x.shape}"
        raw = x
        for adp in self.adapters:
            x = adp(x)

        z = self.aggregate(x, raw)
        return z

if __name__ == "__main__":
    B, T, D = 2, 16, 768
    x = torch.randn(B, T, D)

    model = PTAFusion(
        dim=D,
        num_adapters=2,
        adapter_ratio=0.25,
        kernel_size=3,
        dropout=0.1,
        agg="mlp"
    )
    z = model(x)
    print("PTAFusion:", z.shape)  # [2, 768]