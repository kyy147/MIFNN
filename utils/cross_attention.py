import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Project inputs to query, key, value
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Concatenate heads and project back to original dimension
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        # Cross-attention (x attends to context)
        attn_output = self.cross_attention(x, context, context)
        x = self.norm(x + self.dropout(attn_output))
        return x

class TripleVectorTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Self-attention blocks for each vector
        self.self_attn_a = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.self_attn_b = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.self_attn_c = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        
        # Cross-attention blocks
        self.cross_attn_ab = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attn_ac = CrossAttentionBlock(embed_dim, num_heads, dropout)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, a, b, c):
        # Input shapes: (batch_size, embed_dim)
        
        # Add sequence dimension (batch_size, 1, embed_dim)
        a = a.unsqueeze(1)
        b = b.unsqueeze(1)
        c = c.unsqueeze(1)
        
        # Self-attention for each vector
        a = self.self_attn_a(a) + a
        b = self.self_attn_b(b) + b
        c = self.self_attn_c(c) + c
        
        # Cross-attention between a-b and a-c
        a_ab = self.cross_attn_ab(a, b)  # a attends to b 
        a_ac = self.cross_attn_ac(a, c)  # a attends to c
        # a_ab = self.cross_attn_ab(b, a)  # a attends to b # 使用BUS 和CDFI关注 提取出来的特征很容易过拟合
        # a_ac = self.cross_attn_ac(c, a)  # a attends to c
        
        # Combine representations
        a_combined = a + a_ab + a_ac  # Sum all a representations
        
        # Remove sequence dimension and concatenate
        a_combined = a_combined.squeeze(1)
        b = b.squeeze(1)
        c = c.squeeze(1)
        
        combined = torch.cat([a_combined, b, c], dim=1)
        
        # Final classification
        # output = self.classifier(combined)
        
        return combined

class DualVectorTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Self-attention blocks for each vector
        self.self_attn_a = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.self_attn_b = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        
        # Cross-attention blocks
        self.cross_attn_ab = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attn_ba = CrossAttentionBlock(embed_dim, num_heads, dropout)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, a, b):
        # Input shapes: (batch_size, embed_dim)
        
        # Add sequence dimension (batch_size, 1, embed_dim)
        a = a.unsqueeze(1)
        b = b.unsqueeze(1)
        
        # Self-attention for each vector
        a = self.self_attn_a(a) + a
        b = self.self_attn_b(b) + b
        
        # Cross-attention between a and b (bidirectional)
        a_ab = self.cross_attn_ab(a, b)  # a attends to b
        b_ba = self.cross_attn_ba(b, a)  # b attends to a
        
        # Combine representations
        a_combined = a + a_ab  # a with information from b
        b_combined = b + b_ba  # b with information from a
        
        # Remove sequence dimension and concatenate
        a_combined = a_combined.squeeze(1)
        b_combined = b_combined.squeeze(1)
        
        combined = torch.cat([a_combined, b_combined], dim=1)
        
        # Final classification
        # output = self.classifier(combined)
        
        return combined

class VectorAttentionBottleneck(nn.Module):
    def __init__(self, d_model, num_heads, bottleneck_size):
        super(VectorAttentionBottleneck, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.bottleneck_size = bottleneck_size
        
        # Self-attention for each input vector
        self.self_attn_a = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.self_attn_b = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.self_attn_c = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # Bottleneck layers: project concatenated features to low-dim space
        concat_dim = d_model * 3  # Now three vectors
        self.bottleneck_query = nn.Linear(concat_dim, self.bottleneck_size)
        self.bottleneck_key   = nn.Linear(concat_dim, self.bottleneck_size)
        self.bottleneck_value = nn.Linear(concat_dim, self.bottleneck_size)
        self.bottleneck_out   = nn.Linear(self.bottleneck_size, concat_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化线性层权重
        for module in [self.bottleneck_query, self.bottleneck_key, self.bottleneck_value, self.bottleneck_out]:
            if isinstance(module, nn.Linear):
                # Xavier初始化权重
                nn.init.xavier_uniform_(module.weight)
                # 零初始化偏置
                nn.init.zeros_(module.bias)
        
        # 初始化MultiheadAttention的权重
        for attn_module in [self.self_attn_a, self.self_attn_b, self.self_attn_c]:
            # 初始化in_proj_weight (query, key, value的投影权重)
            if hasattr(attn_module, 'in_proj_weight') and attn_module.in_proj_weight is not None:
                nn.init.xavier_uniform_(attn_module.in_proj_weight)
            
            # 初始化out_proj权重
            if hasattr(attn_module, 'out_proj') and hasattr(attn_module.out_proj, 'weight'):
                nn.init.xavier_uniform_(attn_module.out_proj.weight)
                nn.init.zeros_(attn_module.out_proj.bias)

    def forward(self, x_a, x_b, x_c):
        """
        Args:
            x_a, x_b, x_c: (B, d_model) or (B, L, d_model)
        Returns:
            fused: (B, d_model) or (B, L, d_model) — same shape as input
        """
        # 如果输入是 (B, d_model)，扩展为 (B, 1, d_model)
        if x_a.dim() == 2:
            x_a = x_a.unsqueeze(1)  # (B, 1, d_model)
            x_b = x_b.unsqueeze(1)
            x_c = x_c.unsqueeze(1)

        # Self-attention on each vector
        # output shape: (B, 1, d_model)
        x_a = self.self_attn_a(x_a, x_a, x_a, need_weights=False)[0] + x_a
        x_b = self.self_attn_b(x_b, x_b, x_b, need_weights=False)[0] + x_b
        x_c = self.self_attn_c(x_c, x_c, x_c, need_weights=False)[0] + x_c

        # Concatenate all three vectors along feature dim
        x_concat = torch.cat([x_a, x_b, x_c], dim=-1)  # (B, 1, d_model*3)

        # Project to bottleneck dimension
        q = self.bottleneck_query(x_concat)  # (B, 1, bottleneck_size)
        k = self.bottleneck_key(x_concat)
        v = self.bottleneck_value(x_concat)

        # 简化的注意力机制：直接使用缩放点积注意力
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.bottleneck_size ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重到value
        attended_values = torch.matmul(attention_weights, v)  # (B, 1, bottleneck_size)
        
        # 通过输出投影层
        fused = self.bottleneck_out(attended_values)  # (B, 1, d_model*3)
        fused = fused.squeeze(1)
        
        # 将拼接的特征重新分离并平均
        # batch_size = fused.size(0)
        # fused = fused.view(batch_size, 3, self.d_model)  # (B, 3, d_model)
        # fused = fused.mean(dim=1)  # (B, d_model) - 平均三个特征

        return fused


# Example usage
if __name__ == "__main__":
    # Test TripleVectorTransformer
    print("Testing TripleVectorTransformer:")
    vector_attn = VectorAttentionBottleneck(d_model=768, num_heads=4, bottleneck_size=768)
    
    # Create dummy inputs
    batch_size = 4
    a = torch.randn(batch_size, 768)
    b = torch.randn(batch_size, 768)
    c = torch.randn(batch_size, 768)

    triple_output = vector_attn(a, b, c)
    print(f"TripleVectorTransformer output shape: {triple_output.shape}")  # Should be (batch_size, embed_dim*3)

    # Forward pass
    # triple_output = triple_model(a, b, c)
    # print(f"TripleVectorTransformer output shape: {triple_output.shape}")  # Should be (batch_size, embed_dim*3)
    
    # # Test DualVectorTransformer
    # print("\nTesting DualVectorTransformer:")
    # dual_model = DualVectorTransformer()
    
    # # Forward pass with two vectors
    # dual_output = dual_model(a, b)
    # print(f"DualVectorTransformer output shape: {dual_output.shape}")  # Should be (batch_size, embed_dim*2)
    
    # print(f"Sample triple outputs: {triple_output[0][:5]}")  # Show first 5 elements of first sample
    # print(f"Sample dual outputs: {dual_output[0][:5]}")  # Show first 5 elements of first sample