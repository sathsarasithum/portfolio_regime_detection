"""
Reward Calculation Module
==========================
Computes the net reward for PPO training:
    Net Reward = Sharpe Ratio − Transaction Costs + EVaR Penalty

Components:
    1. Risk-adjusted return (rolling Sharpe ratio)
    2. Transaction cost penalty
    3. EVaR (Entropic Value-at-Risk) penalty for tail risk control
"""

import numpy as np
import torch
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Reward Calculation Module.
    
    Designed to align the RL agent's objective with practical portfolio
    management goals: maximize risk-adjusted returns while controlling
    downside risk and transaction costs.
    
    Net Reward r_t = Sharpe_component − Cost_penalty + EVaR_penalty
    """

    def __init__(
        self,
        risk_free_rate: float = 0.06,      # Sri Lanka ~6% annual
        transaction_cost_rate: float = 0.001,
        sharpe_window: int = 20,
        evar_confidence: float = 0.95,
        evar_penalty_weight: float = 0.1,
        turnover_penalty: float = 0.001,
    ):
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        self.tc_rate = transaction_cost_rate
        self.sharpe_window = sharpe_window
        self.evar_confidence = evar_confidence
        self.evar_penalty_weight = evar_penalty_weight
        self.turnover_penalty = turnover_penalty

        # Running statistics for incremental Sharpe
        self.return_buffer = []

    def reset(self):
        """Reset reward calculator state."""
        self.return_buffer = []

    def compute_sharpe_component(
        self,
        portfolio_return: float,
    ) -> float:
        """
        Compute rolling Sharpe ratio component.
        
        Uses a rolling window of recent returns to estimate
        risk-adjusted performance incrementally.
        """
        self.return_buffer.append(portfolio_return)

        if len(self.return_buffer) < 2:
            return portfolio_return  # Not enough data for Sharpe

        # Use rolling window
        window = self.return_buffer[-self.sharpe_window:]
        excess_returns = np.array(window) - self.daily_rf
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns) + 1e-10

        # Annualized Sharpe
        sharpe = mean_excess / std_excess * np.sqrt(252)

        return sharpe

    def compute_evar_penalty(
        self,
        portfolio_return: float,
    ) -> float:
        """
        Compute Entropic Value-at-Risk (EVaR) penalty.
        
        EVaR is a coherent risk measure that provides an upper bound
        on both VaR and CVaR. It penalizes the agent for taking
        positions that lead to extreme tail losses.
        
        EVaR_α = inf_{z>0} { z^{-1} * ln(E[e^{-z*X}] / α) }
        
        Simplified: we use the exponential penalty on negative returns.
        """
        if len(self.return_buffer) < self.sharpe_window:
            return 0.0

        window = np.array(self.return_buffer[-self.sharpe_window:])
        negative_returns = window[window < 0]

        if len(negative_returns) == 0:
            return 0.0  # No downside risk

        # Simplified EVaR computation
        # Use exponential weighting of losses
        z = 1.0  # Risk aversion parameter
        exp_losses = np.exp(-z * negative_returns)
        alpha = 1 - self.evar_confidence

        evar = (1/z) * np.log(np.mean(exp_losses) / max(alpha, 1e-10))

        return -self.evar_penalty_weight * max(evar, 0)

    def compute_cost_penalty(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
    ) -> float:
        """
        Compute transaction cost and turnover penalty.
        
        Penalizes excessive trading to encourage stable allocations.
        """
        turnover = np.abs(new_weights - old_weights).sum()
        cost_penalty = self.tc_rate * turnover
        turnover_penalty = self.turnover_penalty * turnover ** 2  # Quadratic penalty
        return -(cost_penalty + turnover_penalty)

    def compute_reward(
        self,
        portfolio_return: float,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        info: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Compute the full net reward.
        
        Net Reward = Sharpe Component + EVaR Penalty + Cost Penalty
        
        Args:
            portfolio_return: realized portfolio return this period
            old_weights: previous portfolio weights
            new_weights: new portfolio weights (action)
            info: optional additional info from environment
            
        Returns:
            dict with reward components and total reward
        """
        # Component 1: Sharpe ratio
        sharpe_component = self.compute_sharpe_component(portfolio_return)

        # Component 2: EVaR penalty (tail risk)
        evar_penalty = self.compute_evar_penalty(portfolio_return)

        # Component 3: Cost penalty
        cost_penalty = self.compute_cost_penalty(old_weights, new_weights)

        # Net reward
        net_reward = sharpe_component + evar_penalty + cost_penalty

        return {
            "net_reward": net_reward,
            "sharpe_component": sharpe_component,
            "evar_penalty": evar_penalty,
            "cost_penalty": cost_penalty,
            "portfolio_return": portfolio_return,
        }


class DifferentiableReward(torch.nn.Module):
    """
    Differentiable reward computation for end-to-end gradient flow.
    
    This allows gradients from the reward signal to flow back through
    the actor's weight predictions during joint optimization.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.06,
        tc_rate: float = 0.001,
        evar_weight: float = 0.1,
        turnover_weight: float = 0.001,
    ):
        super().__init__()
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        self.tc_rate = tc_rate
        self.evar_weight = evar_weight
        self.turnover_weight = turnover_weight

    def forward(
        self,
        portfolio_returns: torch.Tensor,
        old_weights: torch.Tensor,
        new_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable reward computation.
        
        Args:
            portfolio_returns: (batch,) realized returns
            old_weights: (batch, n_assets) previous weights
            new_weights: (batch, n_assets) new weights
            
        Returns:
            reward: (batch,) net reward signal
        """
        # Sharpe-like: excess return
        excess_return = portfolio_returns - self.daily_rf

        # Transaction cost penalty
        turnover = (new_weights - old_weights).abs().sum(dim=-1)
        cost_penalty = self.tc_rate * turnover

        # Turnover penalty (quadratic)
        turnover_penalty = self.turnover_weight * turnover ** 2

        # Downside penalty (asymmetric)
        downside = torch.relu(-portfolio_returns)
        evar_penalty = self.evar_weight * downside ** 2

        # Net reward
        reward = excess_return - cost_penalty - turnover_penalty - evar_penalty

        return reward
