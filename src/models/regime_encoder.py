"""
Differentiable Regime Encoder
==============================
Combines VSN+LSTM, Temporal Attention (MHA), and Macro Graph Prior (GNN/GAT)
to produce:
    1. h_temp — Latent Market State for Actor/Critic
    2. Latent Regimes — Soft regime probabilities (Bull, Bear, Sideways)

This is the core innovation: regime detection is learned end-to-end with
the portfolio optimization objective, not as a separate pre-processing step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .vsn_lstm import VSNLSTM
from .temporal_attention import TemporalAttentionBlock
from .macro_graph_prior import MacroGraphPrior


class DifferentiableRegimeEncoder(nn.Module):
    """
    Full Differentiable Regime Encoder.
    
    Architecture:
        x_t → V-VSN + LSTM → Temporal Attention (MHA) → h_temp
                    ↓                                       ↓
           Macro Graph Prior (GNN/GAT)              Regime Classifier
                    ↓                                       ↓
            Graph Embedding ──────────→         Latent Regimes
                                              (Bull/Bear/Sideways)
    
    Key: All components are differentiable and trained end-to-end
    with PPO policy gradients flowing back through the encoder.
    """

    def __init__(
        self,
        n_features: int,
        n_assets: int,
        n_macro: int = 0,
        # VSN + LSTM config
        vsn_hidden_dim: int = 64,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        # MHA config
        mha_num_heads: int = 4,
        mha_dropout: float = 0.1,
        # GNN config
        gnn_hidden_dim: int = 64,
        gnn_num_heads: int = 2,
        gnn_num_layers: int = 2,
        # Regime config
        n_regimes: int = 3,
        latent_state_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_assets = n_assets
        self.n_regimes = n_regimes
        self.latent_state_dim = latent_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # ── Component 1: V-VSN + LSTM ──
        self.vsn_lstm = VSNLSTM(
            n_features=n_features,
            vsn_hidden_dim=vsn_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            dropout=dropout,
        )

        # ── Component 2: Temporal Attention (MHA) ──
        self.temporal_attention = TemporalAttentionBlock(
            d_model=lstm_hidden_dim,
            num_heads=mha_num_heads,
            ffn_dim=lstm_hidden_dim * 2,
            dropout=mha_dropout,
        )

        # ── Component 3: Macro Graph Prior (GNN/GAT) ──
        self.macro_graph = MacroGraphPrior(
            n_assets=n_assets,
            n_macro=n_macro,
            node_feature_dim=lstm_hidden_dim,
            hidden_dim=gnn_hidden_dim,
            num_heads=gnn_num_heads,
            num_layers=gnn_num_layers,
            dropout=dropout,
        )

        # ── Fusion Layer ──
        # Combines temporal attention output with graph prior
        fusion_input_dim = lstm_hidden_dim + gnn_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, latent_state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_state_dim, latent_state_dim),
            nn.LayerNorm(latent_state_dim),
        )

        # ── Regime Classifier ──
        # Soft regime probabilities (differentiable)
        self.regime_classifier = nn.Sequential(
            nn.Linear(latent_state_dim, latent_state_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_state_dim // 2, n_regimes),
        )

        # Temperature parameter for regime softmax (learnable)
        self.regime_temperature = nn.Parameter(torch.tensor(1.0))

    def encode_temporal(
        self,
        x: torch.Tensor,
        lstm_hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Encode temporal features through VSN+LSTM and Temporal Attention.
        
        Args:
            x: (batch, seq_len, n_features) preprocessed temporal context
            lstm_hidden: optional LSTM hidden state for online inference
            
        Returns:
            temporal_repr: (batch, lstm_hidden_dim) temporal representation
            var_weights: (batch, seq_len, n_features) feature importance
            attn_weights: (batch, heads, seq_len, seq_len) temporal attention
            lstm_hidden: updated LSTM hidden state
        """
        # VSN + LSTM
        lstm_out, var_weights, lstm_hidden = self.vsn_lstm(x, lstm_hidden)

        # Temporal Attention
        attn_out, attn_weights = self.temporal_attention(lstm_out)

        # Take the last timestep as the temporal representation
        temporal_repr = attn_out[:, -1, :]  # (B, lstm_hidden_dim)

        return temporal_repr, var_weights, attn_weights, lstm_hidden

    def encode_graph(
        self,
        temporal_repr: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode structural relationships through the Macro Graph Prior.
        
        Creates per-node features from the temporal representation and
        processes them through the GAT.
        
        Args:
            temporal_repr: (batch, lstm_hidden_dim) from temporal encoder
            adj: optional adjacency matrix
            
        Returns:
            graph_embedding: (batch, gnn_hidden_dim)
            node_embeddings: (batch, n_nodes, gnn_hidden_dim)
        """
        B = temporal_repr.shape[0]
        n_nodes = self.n_assets + self.macro_graph.n_macro

        # Create per-node features by broadcasting temporal representation
        # In practice, this could use per-asset features
        node_features = temporal_repr.unsqueeze(1).expand(
            B, n_nodes, -1
        )  # (B, n_nodes, lstm_hidden_dim)

        graph_embedding, node_embeddings = self.macro_graph(
            node_features, adj
        )

        return graph_embedding, node_embeddings

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        lstm_hidden: Optional[Tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the Differentiable Regime Encoder.
        
        Args:
            x: (batch, seq_len, n_features) temporal context window x_t
            adj: optional (n_nodes, n_nodes) graph adjacency
            lstm_hidden: optional LSTM hidden state
            
        Returns:
            dict with:
                - h_temp: (batch, latent_state_dim) latent market state
                - regime_probs: (batch, n_regimes) soft regime probabilities
                - regime_logits: (batch, n_regimes) raw regime logits
                - var_weights: (batch, seq_len, n_features) variable importance
                - attn_weights: temporal attention weights
                - lstm_hidden: updated LSTM state
        """
        # Step 1: Temporal encoding
        temporal_repr, var_weights, attn_weights, lstm_hidden = self.encode_temporal(
            x, lstm_hidden
        )

        # Step 2: Graph prior encoding
        graph_embedding, node_embeddings = self.encode_graph(temporal_repr, adj)

        # Step 3: Fusion — combine temporal and graph representations
        fused = torch.cat([temporal_repr, graph_embedding], dim=-1)
        h_temp = self.fusion(fused)  # (B, latent_state_dim)

        # Step 4: Regime classification (differentiable)
        regime_logits = self.regime_classifier(h_temp)
        regime_probs = F.softmax(
            regime_logits / self.regime_temperature.clamp(min=0.1),
            dim=-1,
        )

        return {
            "h_temp": h_temp,
            "regime_probs": regime_probs,
            "regime_logits": regime_logits,
            "var_weights": var_weights,
            "attn_weights": attn_weights,
            "lstm_hidden": lstm_hidden,
            "node_embeddings": node_embeddings,
        }
