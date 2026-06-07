"""
Critic Network (Value V)
=========================
Estimates the state value V(s) for advantage computation.
Uses Linear layers with scalar output.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CriticNetwork(nn.Module):
    """
    Critic Network (Value V).
    
    Estimates V(h_temp) — the expected cumulative reward from the
    current latent market state. Used for computing advantages in PPO.
    
    Output: Linear → scalar value estimate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        n_regimes: int = 3,
    ):
        super().__init__()

        # Regime conditioning — project to final hidden dim to allow residual addition
        self.regime_embed = nn.Linear(n_regimes, hidden_dims[-1])

        # Build MLP
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

        # Value head — scalar output
        self.value_head = nn.Linear(prev_dim, 1)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize with orthogonal weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h_temp: torch.Tensor,
        regime_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_temp: (batch, latent_dim) latent market state
            regime_probs: (batch, n_regimes) soft regime probabilities
            
        Returns:
            value: (batch,) estimated state value
        """
        features = self.feature_net(h_temp)

        # Regime conditioning
        if regime_probs is not None:
            regime_signal = self.regime_embed(regime_probs)
            features = features + regime_signal

        value = self.value_head(features).squeeze(-1)  # (B,)
        return value
