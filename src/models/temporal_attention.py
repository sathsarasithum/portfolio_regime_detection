"""
Temporal Attention (Multi-Head Attention)
==========================================
Applies multi-head self-attention over the temporal dimension
to capture long-range dependencies and focus on the most relevant
time periods for regime detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class TemporalMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) over the temporal dimension.
    
    Takes LSTM output sequence and applies self-attention to let the model
    attend to important historical patterns across the lookback window.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention.
        
        Args:
            Q: (batch, heads, seq_len, d_k)
            K: (batch, heads, seq_len, d_k)
            V: (batch, heads, seq_len, d_k)
            
        Returns:
            output: (batch, heads, seq_len, d_k)
            attention_weights: (batch, heads, seq_len, seq_len)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model) — LSTM output sequence
            
        Returns:
            output: (batch, seq_len, d_model) — attention-enhanced representation
            attention_weights: (batch, heads, seq_len, seq_len)
        """
        B, T, D = x.shape

        # Add positional encoding
        x = self.pos_encoding(x)

        # Compute Q, K, V
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Self-attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)

        # Residual connection + Layer Norm
        output = self.layer_norm(output + x)

        return output, attn_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalAttentionBlock(nn.Module):
    """
    Full temporal attention block with feed-forward network.
    
    Transformer-style block:
        MHA → Add & Norm → FFN → Add & Norm
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mha = TemporalMultiHeadAttention(d_model, num_heads, dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, heads, seq_len, seq_len)
        """
        # Multi-Head Attention
        attn_out, attn_weights = self.mha(x, mask)

        # Feed-Forward with residual
        ffn_out = self.ffn(attn_out)
        output = self.ffn_norm(ffn_out + attn_out)

        return output, attn_weights
