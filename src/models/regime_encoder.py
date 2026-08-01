"""
Differentiable Regime Encoder
==============================
Combines VSN+LSTM, Temporal Attention (MHA), and Macro Graph Prior (GNN/GAT)
to produce:
    1. h_market — Pooled latent market state for the Critic and Regime Classifier
    2. h_asset  — Per-asset latent state for the Actor (one embedding per asset)
    3. Latent Regimes — Soft regime probabilities (Bull, Bear, Sideways)

This is the core innovation: regime detection is learned end-to-end with
the portfolio optimization objective, not as a separate pre-processing step.

Per-asset processing
---------------------
Each asset's window x_{i,t-L:t} is passed through the SAME temporal backbone
(V-VSN -> LSTM -> Temporal MHA) — channel-independent, shared weights across
assets — producing a genuine per-asset embedding H_t in R^{N x d}. The Macro
Graph Prior (GAT) then operates on these N distinct node embeddings, instead
of a single market-wide vector broadcast to every node.

The critic and regime classifier need a single vector, so H_t is pooled
(mean over assets) into h_market after the graph block. The actor instead
consumes the N per-asset embeddings directly (see ActorNetwork), so it can
condition each asset's allocation on that asset's own evidence.
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
    Full Differentiable Regime Encoder — per-asset variant.

    Architecture:
        x_t (B,N,T,F) → V-VSN + LSTM → Temporal Attention (MHA) → H (B,N,d_lstm)
                                                                        ↓
                                                     Macro Graph Prior (GNN/GAT)
                                                                        ↓
                                                    per-asset fusion → h_asset (B,N,latent)
                                                                        ↓
                                                    mean-pool → h_market (B,latent)
                                                                        ↓
                                                              Regime Classifier

    Key: All components are differentiable and trained end-to-end
    with PPO policy gradients flowing back through the encoder.
    """

    def __init__(
        self,
        n_features_per_asset: int,
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
        self.n_features_per_asset = n_features_per_asset
        self.n_assets = n_assets
        self.n_regimes = n_regimes
        self.latent_state_dim = latent_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # ── Component 1: V-VSN + LSTM (shared weights across assets) ──
        self.vsn_lstm = VSNLSTM(
            n_features=n_features_per_asset,
            vsn_hidden_dim=vsn_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            dropout=dropout,
        )

        # ── Component 2: Temporal Attention (MHA), shared weights ──
        self.temporal_attention = TemporalAttentionBlock(
            d_model=lstm_hidden_dim,
            num_heads=mha_num_heads,
            ffn_dim=lstm_hidden_dim * 2,
            dropout=mha_dropout,
        )

        # ── Component 3: Macro Graph Prior (GNN/GAT) — real per-asset nodes ──
        self.macro_graph = MacroGraphPrior(
            n_assets=n_assets,
            n_macro=n_macro,
            node_feature_dim=lstm_hidden_dim,
            hidden_dim=gnn_hidden_dim,
            num_heads=gnn_num_heads,
            num_layers=gnn_num_layers,
            dropout=dropout,
        )

        # ── Per-asset Fusion Layer ──
        # Combines each asset's temporal embedding with its graph-refined
        # embedding. Applied identically (shared weights) to every asset.
        fusion_input_dim = lstm_hidden_dim + gnn_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, latent_state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_state_dim, latent_state_dim),
            nn.LayerNorm(latent_state_dim),
        )

        # ── Regime Classifier (market-level, operates on pooled state) ──
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
        Encode each asset's temporal window through the shared VSN+LSTM+MHA
        backbone (channel-independent: same weights, no cross-asset mixing
        at this stage).

        Args:
            x: (batch, n_assets, seq_len, n_features_per_asset)
            lstm_hidden: optional LSTM hidden state (batch*n_assets folded)

        Returns:
            H: (batch, n_assets, lstm_hidden_dim) per-asset temporal embedding
            var_weights: (batch, n_assets, seq_len, n_features_per_asset)
            attn_weights: (batch, n_assets, heads, seq_len, seq_len)
            lstm_hidden: updated LSTM state
        """
        B, N, T, F = x.shape
        x_flat = x.reshape(B * N, T, F)

        lstm_out, var_weights, lstm_hidden = self.vsn_lstm(x_flat, lstm_hidden)
        attn_out, attn_weights = self.temporal_attention(lstm_out)
        temporal_repr = attn_out[:, -1, :]                     # (B*N, lstm_hidden_dim)

        H = temporal_repr.reshape(B, N, self.lstm_hidden_dim)
        var_weights = var_weights.reshape(B, N, T, F)
        attn_weights = attn_weights.reshape(
            B, N, attn_weights.shape[1], attn_weights.shape[2], attn_weights.shape[3]
        )
        return H, var_weights, attn_weights, lstm_hidden

    def encode_graph(
        self,
        H: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine per-asset embeddings through the Macro Graph Prior (GAT).

        Args:
            H: (batch, n_assets, lstm_hidden_dim) per-asset temporal embeddings
            adj: optional adjacency matrix

        Returns:
            graph_embedding: (batch, gnn_hidden_dim) pooled graph-level readout
            node_embeddings: (batch, n_assets, gnn_hidden_dim) per-asset, graph-refined
        """
        graph_embedding, node_embeddings = self.macro_graph(H, adj)
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
            x: (batch, n_assets, seq_len, n_features_per_asset) temporal
                context per asset (or (batch, seq_len, n_assets*F) — see note)
            adj: optional (n_nodes, n_nodes) graph adjacency
            lstm_hidden: optional LSTM hidden state

        Returns:
            dict with:
                - h_asset: (batch, n_assets, latent_state_dim) per-asset latent state
                - h_temp / h_market: (batch, latent_state_dim) pooled latent market state
                - regime_probs: (batch, n_regimes) soft regime probabilities
                - regime_logits: (batch, n_regimes) raw regime logits
                - var_weights, attn_weights: per-asset diagnostics
                - lstm_hidden: updated LSTM state
                - node_embeddings: (batch, n_assets, gnn_hidden_dim) graph-refined per-asset state
        """
        if x.dim() == 3:
            # Accept the flat (B, T, N*F) layout too and reshape here, so
            # callers that still hand in flattened windows keep working.
            B, T, NF = x.shape
            x = x.view(B, T, self.n_assets, self.n_features_per_asset).permute(0, 2, 1, 3)

        # Step 1: Temporal encoding — per asset, shared weights
        H, var_weights, attn_weights, lstm_hidden = self.encode_temporal(x, lstm_hidden)

        # Step 2: Graph prior encoding — real per-asset nodes
        graph_embedding, node_embeddings = self.encode_graph(H, adj)

        # Step 3: Per-asset fusion — combine temporal and graph-refined state
        B, N, _ = H.shape
        fused = torch.cat([H, node_embeddings], dim=-1)            # (B, N, lstm+gnn)
        h_asset = self.fusion(fused.reshape(B * N, -1)).reshape(B, N, self.latent_state_dim)

        # Step 4: Pool per-asset states into a single market-level state,
        # used by the critic (needs a scalar V(s)) and the regime classifier
        # (Bull/Bear/Sideways is a market-level concept, not per-asset).
        h_market = h_asset.mean(dim=1)                              # (B, latent_state_dim)

        # Step 5: Regime classification (differentiable)
        regime_logits = self.regime_classifier(h_market)
        regime_probs = F.softmax(
            regime_logits / self.regime_temperature.clamp(min=0.1),
            dim=-1,
        )

        return {
            "h_asset": h_asset,           # (B, N, latent_state_dim) — for the actor
            "h_temp": h_market,           # (B, latent_state_dim) — pooled, for critic/back-compat
            "h_market": h_market,         # alias of h_temp, clearer name
            "regime_probs": regime_probs,
            "regime_logits": regime_logits,
            "var_weights": var_weights,
            "attn_weights": attn_weights,
            "lstm_hidden": lstm_hidden,
            "node_embeddings": node_embeddings,
        }
