import torch
import torch.nn as nn
import torch.nn.functional as F

class HTCM(nn.Module):
    """
    Hemodynamic Temporal Change Module
    Hemodynamic Temporal Change Module (Transformer-free version)
    Remove all Transformer blocks, fix multi-GPU training device mismatch issue
    """
    
    def __init__(self, d_model=768, n_heads=4, n_layers=2, dropout=0.1, fusion_type="concat"):
        super(HTCM, self).__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        # 1. Temporal difference encoding layer (keep unchanged)
        self.diff_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Attention pooling layer (keep unchanged)
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 3. Feature fusion layer (fix multi-GPU device issue!)
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == "add":
            self.fusion = lambda x, y: x + y
        elif fusion_type == "weighted_sum":
            # ✅ Fix 1: Correctly initialize learnable parameters, automatically follow model device
            self.temporal_weight = nn.Parameter(torch.ones(1) * 0.5)
            self.change_weight = nn.Parameter(torch.ones(1) * 0.5)
        elif fusion_type == "mlp":
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
    def forward(self, x):
        B, T, D = x.shape
        
        # === Calculate temporal difference features (keep unchanged)===
        forward_diff = torch.cat([
            torch.zeros(B, 1, D, device=x.device),
            x[:, 1:, :] - x[:, :-1, :]
        ], dim=1)
        
        backward_diff = torch.cat([
            x[:, :-1, :] - x[:, 1:, :],
            torch.zeros(B, 1, D, device=x.device)
        ], dim=1)
        
        diff_features = self.diff_encoder(
            torch.cat([forward_diff, backward_diff], dim=-1)
        )
        
        # === No Transformer, directly use original features ===
        temporal_features = x
        change_features = diff_features
        
        # === Attention pooling (keep unchanged)===
        temporal_weights = self.attention_pool(temporal_features)
        temporal_rep = torch.sum(temporal_features * temporal_weights, dim=1)
        
        change_weights = self.attention_pool(change_features)
        change_rep = torch.sum(change_features * change_weights, dim=1)
        
        # === Feature fusion (✅ Fix 2: Remove lambda, explicit logic, solve multi-GPU issue)===
        if self.fusion_type == "concat":
            combined = torch.cat([temporal_rep, change_rep], dim=-1)
            video_representation = self.fusion(combined)
        elif self.fusion_type == "add":
            video_representation = self.fusion(temporal_rep, change_rep)
        elif self.fusion_type == "weighted_sum":
            # Directly calculate weighted sum here, completely avoid lambda device issue
            video_representation = self.temporal_weight * temporal_rep + self.change_weight * change_rep
        elif self.fusion_type == "mlp":
            combined = torch.cat([temporal_rep, change_rep], dim=-1)
            video_representation = self.fusion(combined)
        
        return video_representation


if __name__ == "__main__":
    model = HTCM(d_model=768, fusion_type='weighted_sum')
    inputs = torch.randn(8, 16, 768)
    outputs = model(inputs)
    print(outputs.shape)