"""
Actor Network (Policy π)
=========================
Maps the per-asset latent state h_asset to portfolio weight allocations.

Each asset's embedding is passed through the SAME per-asset trunk (shared
weights, no cross-asset mixing here — that already happened upstream in the
regime encoder's cross-sectional/graph blocks) to produce one raw logit per
asset. Long-only allocations are then built via softmax (weights sum to 1,
all >= 0); long/short allocations via an independent per-asset tanh (each
asset's position is decided on its own evidence, not forced to trade off
against the others).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math


class ActorNetwork(nn.Module):
    """
    Actor Network (Policy π).

    Takes h_asset (per-asset latent state from the Regime Encoder) and outputs:
        - a_t: Portfolio weights (asset allocations)
        - log_prob: Log probability of the action (for PPO)

    Output activation:
        - Long-only: softmax over assets (weights sum to 1, all >= 0)
        - Long-short: independent per-asset tanh in [-1, 1] (DeePM Eq. 10) —
          each asset's position is NOT coupled to the others; there is no
          budget constraint at this stage. Position-size limits and
          turnover costs are enforced downstream in MarketEnvironment.
    """

    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        allow_short: bool = False,
        n_regimes: int = 3,
        logit_scale: float = 5.0,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.allow_short = allow_short
        # Bound on the pre-softmax/pre-tanh logits. A plain tanh (scale 1.0)
        # caps the achievable concentration at ~1/n_assets * e^2, i.e.
        # near-uniform weights; scale 5 lets softmax express anything from
        # uniform to ~fully concentrated, and gives the long/short tanh room
        # to reach genuinely large (near +-1) positions instead of saturating
        # at a tiny range around 0.
        self.logit_scale = logit_scale

        # Regime conditioning — modulate action based on detected regime.
        # regime_probs is a MARKET-level signal (B, n_regimes); it's broadcast
        # (added) to every asset's per-asset features below.
        self.regime_embed = nn.Linear(n_regimes, hidden_dims[-1])

        # Per-asset trunk. nn.Linear/LayerNorm/ReLU/Dropout all operate on the
        # last dimension, so this Sequential works unchanged whether the
        # input is (B, input_dim) or (B, N, input_dim) — every asset is
        # processed independently through the SAME shared weights.
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.feature_net = nn.Sequential(*layers)

        # Mean head — ONE logit per asset, shared weights across assets
        # (this is what makes the actor genuinely per-asset rather than a
        # single monolithic head that outputs all N weights at once).
        self.mean_head = nn.Linear(prev_dim, 1)

        # Log standard deviation (learnable, for exploration) — one per asset
        self.log_std = nn.Parameter(torch.zeros(n_assets) - 0.5)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Standard PPO initialization: sqrt(2) gain on the ReLU trunk so the
        features actually carry signal, and a small gain on the policy head
        only, so the *initial* policy is near-uniform without crippling the
        gradient through the trunk.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.regime_embed.weight, gain=0.01)
        nn.init.zeros_(self.regime_embed.bias)

    def _mean_logits(
        self,
        h_asset: torch.Tensor,
        regime_probs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Shared per-asset trunk: per-asset latent state (+ broadcast regime)
        -> mean pre-squash logit for each asset.

        Args:
            h_asset: (batch, n_assets, latent_dim) per-asset latent state
            regime_probs: (batch, n_regimes) market-level regime signal

        Returns:
            mean: (batch, n_assets) bounded logits, one per asset
        """
        features = self.feature_net(h_asset)              # (B, N, hidden[-1])

        # Regime conditioning (if available) — broadcast the market-level
        # signal to every asset.
        if regime_probs is not None:
            regime_signal = self.regime_embed(regime_probs)   # (B, hidden[-1])
            features = features + regime_signal.unsqueeze(1)  # (B, N, hidden[-1])

        # Soft-bounded logits: saturates at +-logit_scale but keeps gradient
        raw = self.mean_head(features).squeeze(-1)         # (B, N)
        return self.logit_scale * torch.tanh(raw / self.logit_scale)

    def forward(
        self,
        h_asset: torch.Tensor,
        regime_probs: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_asset: (batch, n_assets, latent_dim) per-asset latent state
            regime_probs: (batch, n_regimes) soft regime probabilities
            deterministic: if True, return mean action (no sampling)

        Returns:
            weights: (batch, n_assets) portfolio weights, for the environment
            raw_action: (batch, n_assets) the Gaussian sample the log_prob
                refers to — this is what PPO must store and re-evaluate
            log_prob: (batch,) log π(raw_action | s)
            entropy: (batch,) policy entropy
        """
        mean = self._mean_logits(h_asset, regime_probs)

        # Standard deviation
        std = self.log_std.exp().expand_as(mean)

        dist = torch.distributions.Normal(mean, std)

        # Sample action from Gaussian policy
        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()  # Reparameterization trick

        # Log prob is of the RAW sample — the quantity PPO's ratio needs.
        log_prob = dist.log_prob(raw_action).sum(dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)              # (B,)

        # Deterministic squash to a valid allocation (not part of the density)
        weights = self._normalize_weights(raw_action)

        return weights, raw_action, log_prob, entropy

    def _normalize_weights(self, raw_weights: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw per-asset weights to a valid portfolio allocation.

        Long-only: softmax normalization (weights sum to 1, all >= 0).
            Assets are COUPLED — raising one weight forces others down.

        Long-short (DeePM Eq. 10): p_i = tanh(a_i), taken directly with NO
            cross-asset renormalization. Each asset's position is decided
            independently — the model can go long every asset, short every
            asset, go flat everywhere, or any mix, without one position
            forcing a change in another. Position-size limits (and hence
            the resulting bound on gross/net exposure) are enforced in
            MarketEnvironment._clip_weights, not here — DeePM instead relies
            on volatility-targeted notional sizing (w = p/sigma_hat) for
            this, which is not yet implemented in this codebase.
        """
        if not self.allow_short:
            # Long-only: use softmax to ensure positive weights summing to 1
            weights = F.softmax(raw_weights, dim=-1)
        else:
            # Long-short: independent per-asset signal, no coupling
            weights = torch.tanh(raw_weights)

        return weights

    def get_log_prob(
        self,
        h_asset: torch.Tensor,
        raw_action: torch.Tensor,
        regime_probs: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability of a given action (for PPO ratio).

        Args:
            h_asset: (batch, n_assets, latent_dim) per-asset latent state
            raw_action: (batch, n_assets) the *pre-squash* Gaussian sample
                stored during rollout — NOT the normalized portfolio weights
            regime_probs: (batch, n_regimes)

        Returns:
            log_prob: (batch,)
            entropy: (batch,)
        """
        mean = self._mean_logits(h_asset, regime_probs)
        std = self.log_std.exp().expand_as(mean)

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy
