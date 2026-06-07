"""
Generalized Advantage Estimation (GAE)
========================================
Computes advantages for PPO using the GAE(λ) algorithm.
"""

import torch
import numpy as np
from typing import Tuple


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE(λ) = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    Args:
        rewards: (T,) rewards from environment
        values: (T,) value estimates from critic
        dones: (T,) episode termination flags
        next_value: scalar, V(s_{T+1})
        gamma: discount factor
        gae_lambda: GAE lambda for bias-variance tradeoff
        
    Returns:
        advantages: (T,) GAE advantages
        returns: (T,) discounted returns (advantages + values)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_val = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        # TD error
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]

        # GAE
        advantages[t] = last_gae = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae
        )

    returns = advantages + values

    return advantages, returns


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Normalize advantages for stable training."""
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
