"""
PPO Agent
==========
Combines the Differentiable Regime Encoder, Actor, and Critic into
a single jointly-optimized PPO agent.

This is the top-level model that implements the full forward pass
and joint end-to-end gradient update.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .regime_encoder import DifferentiableRegimeEncoder
from .actor import ActorNetwork
from .critic import CriticNetwork


class PPOAgent(nn.Module):
    """
    Jointly Optimized PPO Agent for Portfolio Management.
    
    Architecture:
        x_t → Differentiable Regime Encoder → h_temp, regime_probs
                        ↓                          ↓
                  Actor Network (π)           Critic Network (V)
                        ↓                          ↓
                  a_t (weights)              V(s) (value estimate)
    
    Joint End-to-End Gradient Update:
        All parameters (encoder + actor + critic) are updated together
        via PPO's clipped objective + value loss + entropy bonus.
    """

    def __init__(
        self,
        n_features_per_asset: int,
        n_assets: int,
        n_macro: int = 0,
        # Regime encoder config
        vsn_hidden_dim: int = 64,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        mha_num_heads: int = 4,
        gnn_hidden_dim: int = 64,
        gnn_num_heads: int = 2,
        gnn_num_layers: int = 2,
        n_regimes: int = 3,
        latent_state_dim: int = 128,
        # Actor/Critic config
        actor_hidden_dims: list = None,
        critic_hidden_dims: list = None,
        encoder_dropout: float = 0.1,
        mha_dropout: float = 0.1,
        actor_dropout: float = 0.1,
        critic_dropout: float = 0.1,
        allow_short: bool = False,
    ):
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128]

        self.n_assets = n_assets
        self.n_regimes = n_regimes
        self.latent_state_dim = latent_state_dim

        # ── Differentiable Regime Encoder ──
        self.regime_encoder = DifferentiableRegimeEncoder(
            n_features_per_asset=n_features_per_asset,
            n_assets=n_assets,
            n_macro=n_macro,
            vsn_hidden_dim=vsn_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            mha_num_heads=mha_num_heads,
            mha_dropout=mha_dropout,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_heads=gnn_num_heads,
            gnn_num_layers=gnn_num_layers,
            n_regimes=n_regimes,
            latent_state_dim=latent_state_dim,
            dropout=encoder_dropout,
        )

        # ── Actor Network (Policy π) ──
        self.actor = ActorNetwork(
            input_dim=latent_state_dim,
            n_assets=n_assets,
            hidden_dims=actor_hidden_dims,
            dropout=actor_dropout,
            allow_short=allow_short,
            n_regimes=n_regimes,
        )

        # ── Critic Network (Value V) ──
        self.critic = CriticNetwork(
            input_dim=latent_state_dim,
            hidden_dims=critic_hidden_dims,
            dropout=critic_dropout,
            n_regimes=n_regimes,
        )

    def forward(
        self,
        observation: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        lstm_hidden: Optional[Tuple] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: Observation → Encoder → Actor + Critic.
        
        Args:
            observation: (batch, n_assets, seq_len, n_features_per_asset), or
                (batch, seq_len, n_assets*n_features_per_asset) — the encoder
                reshapes the flat form automatically.
            adj: optional graph adjacency matrix
            lstm_hidden: optional LSTM hidden state
            deterministic: if True, use mean action (no exploration)

        Returns:
            dict with all outputs for PPO training
        """
        # Step 1: Encode observation → per-asset latent state + pooled
        # market state + regime
        encoder_out = self.regime_encoder(observation, adj, lstm_hidden)
        h_asset = encoder_out["h_asset"]      # (B, N, latent_dim) — for the actor
        h_market = encoder_out["h_market"]    # (B, latent_dim) — for the critic
        regime_probs = encoder_out["regime_probs"]

        # Step 2: Actor — generate portfolio weights, per asset
        action, raw_action, log_prob, entropy = self.actor(
            h_asset, regime_probs, deterministic
        )

        # Step 3: Critic — estimate state value from the pooled market state
        value = self.critic(h_market, regime_probs)

        return {
            # Actions & policy
            "action": action,           # (B, n_assets) portfolio weights (→ env)
            "raw_action": raw_action,   # (B, n_assets) pre-softmax sample (→ PPO)
            "log_prob": log_prob,       # (B,) log π(a|s)
            "entropy": entropy,         # (B,) policy entropy
            # Value
            "value": value,             # (B,) V(s)
            # Encoder outputs
            "h_asset": h_asset,         # (B, N, latent_dim) per-asset latent state
            "h_temp": h_market,         # (B, latent_dim) pooled latent state (back-compat name)
            "h_market": h_market,       # (B, latent_dim) pooled latent state
            "regime_probs": regime_probs,  # (B, n_regimes) regime probs
            "regime_logits": encoder_out["regime_logits"],
            "var_weights": encoder_out["var_weights"],
            "attn_weights": encoder_out["attn_weights"],
            "lstm_hidden": encoder_out["lstm_hidden"],
        }

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        raw_action: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate log probability and value of given state-action pairs.
        Used during PPO update to compute the importance sampling ratio.

        Args:
            observation: (batch, n_assets, seq_len, n_features_per_asset), or
                (batch, seq_len, n_assets*n_features_per_asset)
            raw_action: (batch, n_assets) the pre-softmax samples stored during
                rollout. Passing normalized weights here silently breaks the
                PPO ratio — the density is defined over the raw space.

        Returns:
            dict with log_prob, value, entropy for PPO loss computation
        """
        # Encode
        encoder_out = self.regime_encoder(observation, adj)
        h_asset = encoder_out["h_asset"]
        h_market = encoder_out["h_market"]
        regime_probs = encoder_out["regime_probs"]

        # Actor — evaluate given action
        log_prob, entropy = self.actor.get_log_prob(
            h_asset, raw_action, regime_probs
        )

        # Critic — value estimate
        value = self.critic(h_market, regime_probs)

        return {
            "log_prob": log_prob,
            "value": value,
            "entropy": entropy,
            "regime_probs": regime_probs,
        }

    def get_action(
        self,
        observation: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        lstm_hidden: Optional[Tuple] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Convenience method for inference / environment stepping.
        
        Returns:
            action: (n_assets,) portfolio weights
            info: dict with additional information
        """
        with torch.no_grad():
            out = self.forward(observation, adj, lstm_hidden, deterministic)

        return out["action"], {
            "value": out["value"],
            "raw_action": out["raw_action"],
            "log_prob": out["log_prob"],
            "regime_probs": out["regime_probs"],
            "lstm_hidden": out["lstm_hidden"],
        }

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per component."""
        encoder_params = sum(
            p.numel() for p in self.regime_encoder.parameters() if p.requires_grad
        )
        actor_params = sum(
            p.numel() for p in self.actor.parameters() if p.requires_grad
        )
        critic_params = sum(
            p.numel() for p in self.critic.parameters() if p.requires_grad
        )
        total = encoder_params + actor_params + critic_params

        return {
            "regime_encoder": encoder_params,
            "actor": actor_params,
            "critic": critic_params,
            "total": total,
        }
