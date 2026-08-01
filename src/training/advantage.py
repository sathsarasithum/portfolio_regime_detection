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


class RunningMeanStd:
    """Welford running mean/variance estimator (Chan's parallel update)."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = int(x.shape[0])

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))


class RewardNormalizer:
    """
    Scales rewards by a running estimate of the discounted-return std.

    The reward here is an *annualized* Sharpe ratio (x sqrt(252)), so raw
    rewards land around +/-15 and discounted returns reach the hundreds. The
    resulting value loss (~1e4) produces a critic gradient orders of magnitude
    larger than the actor's, and since `clip_grad_norm_` is applied to *all*
    parameters jointly it rescales the whole update — leaving the actor with an
    effective learning rate near zero. Dividing rewards (not mean-centring, so
    the sign of the Sharpe signal is preserved) keeps returns O(1) and the two
    gradients comparable.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.rms = RunningMeanStd()
        self.gamma = gamma
        self.epsilon = epsilon
        self.ret_acc = 0.0

    def reset(self):
        """Reset the discounted-return accumulator at episode boundaries."""
        self.ret_acc = 0.0

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Update statistics from a rollout and return the scaled rewards."""
        discounted = np.empty_like(rewards, dtype=np.float64)
        for t, (r, d) in enumerate(zip(rewards, dones)):
            self.ret_acc = self.ret_acc * self.gamma + float(r)
            discounted[t] = self.ret_acc
            if d:
                self.ret_acc = 0.0

        self.rms.update(discounted)
        return rewards / (self.rms.std + self.epsilon)
