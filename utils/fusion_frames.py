import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# common utils
# =========================

def masked_mean(x, mask=None, dim=1, eps=1e-6):
    """
    x:    [B, T, D]
    mask: [B, T] bool, True means valid
    """
    if mask is None:
        return x.mean(dim=dim)
    w = mask.float().unsqueeze(-1)  # [B, T, 1]
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(eps)


class LearnableTemporalPE(nn.Module):
    def __init__(self, d_model, max_len=32, dropout=0.0):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}")
        T = x.size(1)
        if T > self.pe.size(1):
            raise ValueError(f"T={T} > max_len={self.pe.size(1)}")
        return self.drop(x + self.pe[:, :T])


class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class VideoClassifier(nn.Module):
    """
    Wrapper for CLS-only fusion modules.
    Input:
        x:    [B, T, D]
        mask: [B, T] bool or None
    Output:
        logits: [B, num_classes]
    """
    def __init__(self, fusion, d_model=768, num_classes=100, dropout=0.1):
        super().__init__()
        self.fusion = fusion
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, mask=None):
        feat = self.fusion(x, mask=mask)   # [B, D]
        logits = self.cls_head(feat)
        return logits


class VideoPatchClassifier(nn.Module):
    """
    Wrapper for patch-token fusion modules.
    Input:
        frame_cls:    [B, T, D]
        patch_tokens: [B, T, N, D]
        mask:         [B, T] bool or None
    Output:
        logits: [B, num_classes]
    """
    def __init__(self, fusion, d_model=768, num_classes=100, dropout=0.1):
        super().__init__()
        self.fusion = fusion
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, frame_cls, patch_tokens, mask=None):
        feat = self.fusion(frame_cls=frame_cls, patch_tokens=patch_tokens, mask=mask)
        logits = self.cls_head(feat)
        return logits


# =========================
# 0) Mean Pool baseline
# =========================

class MeanPoolFusion(nn.Module):
    """
    Strong and fair baseline for CLS-only comparison.
    x: [B, T, D]
    return: [B, D]
    """
    def __init__(self, d_model=768, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}")
        feat = masked_mean(self.norm(x), mask, dim=1)
        return self.drop(feat)


# =========================
# 1) DejaVid-style
# per-timestep, per-feature weighting
# =========================

class DejaVidStyleFusion(nn.Module):
    """
    Simplified plug-and-play version:
    - temporal positional embedding
    - lightweight temporal mixing
    - per-timestep, per-feature gate
    - weighted aggregation

    Input:
        x: [B, T, D]
    Output:
        [B, D]
    """
    def __init__(self, d_model=768, max_len=32, hidden_ratio=2, dropout=0.1):
        super().__init__()
        self.pos = LearnableTemporalPE(d_model, max_len=max_len, dropout=dropout)

        # temporal mixing over T
        self.temporal_mix = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

        hidden = d_model * hidden_ratio
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, mask=None):
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}")

        h = self.pos(x)  # [B, T, D]
        h = h + self.temporal_mix(h.transpose(1, 2)).transpose(1, 2)

        # [B, T, D], one gate per timestep and per feature dimension
        w = self.gate(h)

        if mask is not None:
            w = w * mask.float().unsqueeze(-1)

        feat = (w * h).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
        feat = self.out(feat)
        return feat


# =========================
# 2) EVL-style
# learned query + lightweight decoder
# =========================

class EVLDecoderBlock(nn.Module):
    def __init__(self, d_model=768, num_heads=8, mlp_ratio=4, dropout=0.1, kernel_size=3):
        super().__init__()

        # local temporal module on memory
        self.temporal_dw = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.temporal_pw = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_m = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_q2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, q, memory, mask=None):
        """
        q:      [B, 1, D]
        memory: [B, T, D]
        mask:   [B, T] bool or None
        """
        memory = memory + self.temporal_pw(self.temporal_dw(memory.transpose(1, 2))).transpose(1, 2)

        key_padding_mask = None if mask is None else ~mask.bool()  # True means ignore
        attn_out, _ = self.cross_attn(
            query=self.norm_q1(q),
            key=self.norm_m(memory),
            value=self.norm_m(memory),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q = q + attn_out
        q = q + self.mlp(self.norm_q2(q))
        return q, memory


class EVLStyleQueryFusion(nn.Module):
    """
    Simplified EVL-style fusion for CLS-only input.
    Input:
        x: [B, T, D]
    Output:
        [B, D]
    """
    def __init__(self, d_model=768, max_len=32, num_heads=8, num_layers=2, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.pos = LearnableTemporalPE(d_model, max_len=max_len, dropout=dropout)

        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.layers = nn.ModuleList([
            EVLDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}")

        memory = self.pos(x)  # [B, T, D]
        q = self.query.expand(x.size(0), -1, -1)  # [B, 1, D]

        for layer in self.layers:
            q, memory = layer(q, memory, mask=mask)

        feat = self.out_norm(q[:, 0])  # [B, D]
        return feat


# =========================
# 3) TD4V-inspired
# temporal difference on CLS sequence
# =========================

class TD4VInspiredDiffFusion(nn.Module):
    """
    For CLS-only input:
    - compute temporal differences
    - fuse x, dx, |dx|
    - temporal conv
    - attention pooling

    Input:
        x: [B, T, D]
    Output:
        [B, D]
    """
    def __init__(self, d_model=768, hidden_dim=None, max_len=32, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_model

        self.pos = LearnableTemporalPE(d_model, max_len=max_len, dropout=dropout)
        self.in_proj = nn.Linear(d_model * 3, hidden_dim)

        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.score = nn.Linear(hidden_dim, 1)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x, mask=None):
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}")

        x = self.pos(x)

        dx = torch.zeros_like(x)
        dx[:, 1:] = x[:, 1:] - x[:, :-1]

        z = torch.cat([x, dx, dx.abs()], dim=-1)   # [B, T, 3D]
        z = self.in_proj(z)                         # [B, T, H]
        z = z + self.temporal(z.transpose(1, 2)).transpose(1, 2)

        score = self.score(z).squeeze(-1)          # [B, T]
        if mask is not None:
            score = score.masked_fill(~mask.bool(), float("-inf"))

        attn = torch.softmax(score, dim=1).unsqueeze(-1)  # [B, T, 1]
        feat = (attn * z).sum(dim=1)                      # [B, H]
        feat = self.out(feat)                             # [B, D]
        return feat


# =========================
# 4) TC-style context aggregation
# patch-token version
# =========================

class TCStyleContextFusion(nn.Module):
    """
    Patch-token version:
    1) use frame CLS to score informative patch tokens in each frame
    2) select top-k tokens per frame
    3) summarize them into context tokens
    4) use one video query to aggregate final video feature

    Inputs:
        frame_cls:    [B, T, D]
        patch_tokens: [B, T, N, D]
    Output:
        [B, D]
    """
    def __init__(
        self,
        d_model=768,
        max_len=32,
        num_heads=8,
        num_context_tokens=4,
        topk=8,
        mlp_ratio=4,
        dropout=0.1,
        use_cls_memory=True,
    ):
        super().__init__()
        self.topk = topk
        self.use_cls_memory = use_cls_memory

        self.frame_pe = nn.Parameter(torch.zeros(1, max_len, 1, d_model))
        nn.init.trunc_normal_(self.frame_pe, std=0.02)

        self.context_tokens = nn.Parameter(torch.zeros(1, num_context_tokens, d_model))
        self.video_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.context_tokens, std=0.02)
        nn.init.trunc_normal_(self.video_query, std=0.02)

        self.norm_ctx_q = nn.LayerNorm(d_model)
        self.norm_tok = nn.LayerNorm(d_model)
        self.context_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ctx_ffn = nn.LayerNorm(d_model)
        self.context_mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

        self.norm_vq = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(d_model)
        self.final_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_vq_ffn = nn.LayerNorm(d_model)
        self.final_mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, frame_cls, patch_tokens, mask=None):
        if frame_cls.dim() != 3:
            raise ValueError(f"frame_cls must be [B, T, D], got {tuple(frame_cls.shape)}")
        if patch_tokens.dim() != 4:
            raise ValueError(f"patch_tokens must be [B, T, N, D], got {tuple(patch_tokens.shape)}")

        B, T, N, D = patch_tokens.shape
        if T > self.frame_pe.size(1):
            raise ValueError(f"T={T} > max_len={self.frame_pe.size(1)}")

        k = min(self.topk, N)

        frame_pe = self.frame_pe[:, :T]       # [1, T, 1, D]
        cls = frame_cls + frame_pe.squeeze(2) # [B, T, D]
        patches = patch_tokens + frame_pe     # [B, T, N, D]

        # score patch tokens by similarity to frame CLS
        cls_n = F.normalize(cls, dim=-1).unsqueeze(2)     # [B, T, 1, D]
        patches_n = F.normalize(patches, dim=-1)          # [B, T, N, D]
        score = (cls_n * patches_n).sum(dim=-1)           # [B, T, N]

        # optionally invalidate padded frames
        if mask is not None:
            score = score.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))

        topk_idx = score.topk(k=k, dim=2).indices         # [B, T, k]
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        selected = torch.gather(patches, dim=2, index=gather_idx)  # [B, T, k, D]
        tokens = selected.reshape(B, T * k, D)            # [B, T*k, D]

        # summarize into context tokens
        context = self.context_tokens.expand(B, -1, -1)   # [B, M, D]
        ctx_out, _ = self.context_attn(
            query=self.norm_ctx_q(context),
            key=self.norm_tok(tokens),
            value=self.norm_tok(tokens),
            need_weights=False,
        )
        context = context + ctx_out
        context = context + self.context_mlp(self.norm_ctx_ffn(context))

        # final memory can be context only, or [context + frame_cls]
        memory = context if not self.use_cls_memory else torch.cat([context, cls], dim=1)

        q = self.video_query.expand(B, -1, -1)            # [B, 1, D]
        q_out, _ = self.final_attn(
            query=self.norm_vq(q),
            key=self.norm_mem(memory),
            value=self.norm_mem(memory),
            need_weights=False,
        )
        q = q + q_out
        q = q + self.final_mlp(self.norm_vq_ffn(q))

        feat = self.out_norm(q[:, 0])                     # [B, D]
        return feat


if __name__ == "__main__":
    B, T, N, D = 2, 16, 200, 768
    cls_token = torch.randn(B, T, D)
    all_tokens = torch.randn(B, T, N, D)

    model = TCStyleContextFusion(
        d_model=D
    )
    z = model(cls_token, all_tokens)
    print("MCTFFusion:", z.shape)  # [2, 768]