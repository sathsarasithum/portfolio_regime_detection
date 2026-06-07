"""
PyTorch Dataset for Portfolio RL Training
==========================================
Provides windowed observations for the PPO agent.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional


class PortfolioDataset(Dataset):
    """
    Dataset that provides temporal context windows for the RL environment.
    
    Each sample contains:
        - observation: (window_size, n_features) temporal context x_t
        - prices: (window_size, n_assets) raw prices for return calculation
        - date_idx: index into the date array
    """

    def __init__(
        self,
        windows: np.ndarray,
        prices: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ):
        """
        Args:
            windows: (N, window_size, n_features) preprocessed feature windows
            prices: (N, n_assets) raw closing prices aligned with window end dates
            returns: (N, n_assets) log returns aligned with window end dates
        """
        self.windows = torch.FloatTensor(windows)
        self.prices = torch.FloatTensor(prices)
        self.returns = torch.FloatTensor(returns) if returns is not None else None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "observation": self.windows[idx],     # (window_size, n_features)
            "prices": self.prices[idx],            # (n_assets,)
            "date_idx": idx,
        }
        if self.returns is not None:
            sample["returns"] = self.returns[idx]  # (n_assets,)
        return sample


class RolloutBuffer:
    """
    Stores rollout data for PPO updates.
    
    Collects:
        - observations, actions, log_probs, values, rewards, dones
        - Computes returns and advantages via GAE
    """

    def __init__(self, buffer_size: int, obs_shape: tuple,
                 action_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False

        # Pre-allocate tensors
        self.observations = torch.zeros((buffer_size, *obs_shape), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)

    def add(self, obs, action, log_prob, value, reward, done):
        """Add a single transition to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True

    def compute_gae(self, last_value: float, gamma: float = 0.99,
                    gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation."""
        size = self.buffer_size if self.full else self.ptr
        last_gae = 0.0

        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - 0.0  # Last step
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t + 1]

            delta = (self.rewards[t] + gamma * next_value * next_non_terminal
                     - self.values[t])
            self.advantages[t] = last_gae = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae
            )

        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def get_batches(self, batch_size: int):
        """Yield mini-batches for PPO update."""
        size = self.buffer_size if self.full else self.ptr
        indices = torch.randperm(size, device=self.device)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_idx = indices[start:end]

            yield {
                "observations": self.observations[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "values": self.values[batch_idx],
                "returns": self.returns[batch_idx],
                "advantages": self.advantages[batch_idx],
            }

    def reset(self):
        """Reset buffer for new rollout."""
        self.ptr = 0
        self.full = False
