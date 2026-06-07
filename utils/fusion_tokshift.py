import torch
import torch.nn as nn

class TokShiftFusion(nn.Module):
    """
    TokShift-style fusion on [CLS] features.

    Input:  x [B, T, D]
    Output: z [B, D]

    Args:
        shift_ratio: ratio of channels used for temporal shift
        agg: one of ["mean", "max", "attn"]
    """
    def __init__(self, dim: int = 768, shift_ratio: float = 0.2, agg: str = "attn", dropout: float = 0.1):
        super().__init__()
        assert 0.0 < shift_ratio < 0.8, "shift_ratio should be in (0, 0.8)"
        assert agg in ["mean", "max", "attn", "mlp"]

        self.dim = dim
        self.shift_ratio = shift_ratio
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
    def temporal_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        k = int(D * self.shift_ratio)

        out = x.clone()

        # first k channels: shift forward (use previous frame)
        if k > 0:
            out[:, 1:, :k] = x[:, :-1, :k]
            out[:, 0, :k] = x[:, 0, :k]

            # last k channels: shift backward (use next frame)
            out[:, :-1, -k:] = x[:, 1:, -k:]
            out[:, -1, -k:] = x[:, -1, -k:]

        return out

    def aggregate(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        if self.agg == "mean":
            return x.mean(dim=1)
        elif self.agg == "max":
            return x.max(dim=1).values
        elif self.agg == "attn":
            score = self.attn_score(x)          # [B, T, 1]
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
        y = self.temporal_shift(x)
        z = self.aggregate(y, raw)
        return z


if __name__ == "__main__":
    B, T, D = 2, 16, 768
    x = torch.randn(B, T, D)

    model = TokShiftFusion(dim=D, shift_ratio=0.25, agg="mlp")
    z = model(x)
    print("TokShiftFusion:", z.shape)  # [2, 768]