# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import ml_collections

import torch
import torch.nn as nn

from torch.nn import Dropout, Softmax, Linear, LayerNorm
import torch.nn.functional as F

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):  # B (n + 1) all_head_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # B (n + 1) num_attention_heads attention_head_size
        return x.permute(0, 2, 1, 3)  # B num_attention_heads (n + 1) attention_head_size

    def forward(self, hidden_states):  # B (n + 1) h
        mixed_query_layer = self.query(hidden_states)  # B (n + 1) all_head_size
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # B num_attention_heads (n + 1) attention_head_size

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)  # B num_attention_heads (n + 1) (n + 1)

        context_layer = torch.matmul(attention_probs, value_layer)  # B num_attention_heads (n + 1) attention_head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # B (n + 1) num_attention_heads attention_head_size
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # B (n + 1) all_head_size
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config, init=True):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        if init:
            self._init_weights()
        # self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, init=True):
        super(Embeddings, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # self.modalityFeatureEncoding = nn.Parameter(torch.zeros(1, 3, config.hidden_size))
        if init:
            self._init_weights()

    def forward(self, x):  # B * 2 * d(hidden_num=128)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1) # -1 means keep the dimension unchanged
        x = torch.cat((cls_tokens, x), dim=1) # B (n + 1) h
        # x = x + self.modalityFeatureEncoding
        return x
    
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x): # B (n + 1) h
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Module):
    def __init__(self, config, init=True):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        if init:
            self._initialize_weights()

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded
    
    def _initialize_weights(self):
        """Initialize weights for the model."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm,)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)


def get_transformer_config(out_dim=768,num_heads=4, num_layers=4):
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.hidden_size = out_dim
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = out_dim // 4
    config.transformer.num_heads = num_heads  # 4000
    config.transformer.num_layers = num_layers
    config.transformer.attention_dropout_rate = 0.3
    config.transformer.dropout_rate = 0.3
    config.representation_size = None
    return config


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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize linear layer weights
        for module in [self.bottleneck_query, self.bottleneck_key, self.bottleneck_value, self.bottleneck_out]:
            if isinstance(module, nn.Linear):
                # Xavier initialization for weights
                nn.init.xavier_uniform_(module.weight)
                # Zero initialization for bias
                nn.init.zeros_(module.bias)
        
        # Initialize MultiheadAttention weights
        for attn_module in [self.self_attn_a, self.self_attn_b, self.self_attn_c]:
            # Initialize in_proj_weight (projection weights for query, key, value)
            if hasattr(attn_module, 'in_proj_weight') and attn_module.in_proj_weight is not None:
                nn.init.xavier_uniform_(attn_module.in_proj_weight)
            
            # Initialize out_proj weights
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
        # If input is (B, d_model), expand to (B, 1, d_model)
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

        # Simplified attention mechanism: directly use scaled dot-product attention
        # Calculate attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.bottleneck_size ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to value
        attended_values = torch.matmul(attention_weights, v)  # (B, 1, bottleneck_size)
        
        # Pass through output projection layer
        fused = self.bottleneck_out(attended_values)  # (B, 1, d_model*3)
        
        # Split the concatenated features and average them
        # batch_size = fused.size(0)
        # fused = fused.view(batch_size, 3, self.d_model)  # (B, 3, d_model)
        # fused = fused.mean(dim=1)  # (B, d_model) - average three features

        return fused

def main():
    # Assume output dimension is 256
    output_dim = 768
    config = get_transformer_config(output_dim)

    # Instantiate Transformer model
    model = Transformer(config)

    # Generate random input data, assuming batch size is 1, sequence length is 10
    input1 = torch.rand(32, 1, config.hidden_size)
    input2 = torch.rand(32, 1, config.hidden_size)
    input3 = torch.rand(32, 1, config.hidden_size)
    input = torch.cat([input1, input2, input3], dim=1)
    # input_ids = torch.cat([input1, input2], dim=1)
    print(model)

    # Pass input through model
    output = model(input)
    print(output.shape)

    classifiator = nn.Sequential(
        nn.Linear(output_dim, 2),
        nn.Softmax(-1)
    )

    output = classifiator(output[:, 0, :])

    # Output results
    print("Output shape:", output.shape)
    # print("Output tensor:", output.shape)


if __name__ == "__main__":
    main()