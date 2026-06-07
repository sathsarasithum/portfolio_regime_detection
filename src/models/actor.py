"""
Actor Network (Policy π)
=========================
Maps the latent market state h_temp to portfolio weight allocations.
Uses Linear layers with tanh output to produce weights in [-1, 1],
then normalizes to a valid portfolio allocation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math


class ActorNetwork(nn.Module):
    """
    Actor Network (Policy π).
    
    Takes h_temp (latent market state from Regime Encoder) and outputs:
        - a_t: Portfolio weights (asset allocations)
        - log_prob: Log probability of the action (for PPO)
    
    Output activation: tanh → normalized to sum to 1 (long-only)
    or kept in [-1, 1] range (if short selling is allowed).
    """

    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        allow_short: bool = False,
        n_regimes: int = 3,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.allow_short = allow_short

        # Regime conditioning — modulate action based on detected regime
        # Project to final hidden dim (output of feature_net) to allow residual addition
        self.regime_embed = nn.Linear(n_regimes, hidden_dims[-1])

        # Build MLP layers
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

        # Mean head (portfolio weights)
        self.mean_head = nn.Linear(prev_dim, n_assets)

        # Log standard deviation (learnable, for exploration)
        self.log_std = nn.Parameter(torch.zeros(n_assets) - 0.5)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable initial policy."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h_temp: torch.Tensor,
        regime_probs: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_temp: (batch, latent_dim) latent market state
            regime_probs: (batch, n_regimes) soft regime probabilities
            deterministic: if True, return mean action (no sampling)
            
        Returns:
            action: (batch, n_assets) portfolio weights
            log_prob: (batch,) log probability of the action
            entropy: (batch,) policy entropy
        """
        # Feature extraction
        features = self.feature_net(h_temp)

        # Regime conditioning (if available)
        if regime_probs is not None:
            regime_signal = self.regime_embed(regime_probs)
            features = features + regime_signal

        # Mean portfolio weights
        mean = torch.tanh(self.mean_head(features))  # (B, n_assets), range [-1, 1]

        # Standard deviation
        std = self.log_std.exp().expand_as(mean)

        # Sample action from Gaussian policy
        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()  # Reparameterization trick

        # Compute log probability
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)           # (B,)

        # Normalize to valid portfolio weights
        action = self._normalize_weights(action)

        return action, log_prob, entropy

    def _normalize_weights(self, raw_weights: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw weights to valid portfolio allocations.
        
        Long-only: softmax normalization (weights sum to 1, all >= 0)
        Long-short: normalize by L1 norm
        """
        if not self.allow_short:
            # Long-only: use softmax to ensure positive weights summing to 1
            weights = F.softmax(raw_weights, dim=-1)
        else:
            # Long-short: normalize by L1 norm
            weights = raw_weights / (raw_weights.abs().sum(dim=-1, keepdim=True) + 1e-8)

        return weights

    def get_log_prob(
        self,
        h_temp: torch.Tensor,
        action: torch.Tensor,
        regime_probs: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability of a given action (for PPO ratio).
        
        Args:
            h_temp: (batch, latent_dim) latent state
            action: (batch, n_assets) action to evaluate
            regime_probs: (batch, n_regimes)
            
        Returns:
            log_prob: (batch,)
            entropy: (batch,)
        """
        features = self.feature_net(h_temp)
        if regime_probs is not None:
            regime_signal = self.regime_embed(regime_probs)
            features = features + regime_signal

        mean = torch.tanh(self.mean_head(features))
        std = self.log_std.exp().expand_as(mean)

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy
