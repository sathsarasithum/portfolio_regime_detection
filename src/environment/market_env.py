"""
Market Environment
===================
Simulates the trading environment for portfolio management.
Executes trades, tracks portfolio value, and calculates transaction costs.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MarketEnvironment:
    """
    Market Environment for Portfolio RL.
    
    Simulates:
        1. Executing trades based on actor's weight allocations
        2. Computing realized returns
        3. Tracking portfolio value over time
        4. Computing transaction costs from weight changes
    
    State:
        - Current portfolio weights
        - Portfolio value history
        - Transaction cost accumulator
    """

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_value: float = 1_000_000.0,
        transaction_cost_rate: float = 0.001,
        slippage: float = 0.0005,
        allow_short: bool = False,
        max_position_size: float = 0.30,
    ):
        """
        Args:
            prices: (T, n_assets) closing prices
            features: (T, window_size, n_features) preprocessed observation windows
            initial_value: initial portfolio value in LKR
            transaction_cost_rate: proportional transaction cost
            slippage: execution slippage rate
            allow_short: whether short selling is allowed
            max_position_size: maximum weight in single asset
        """
        self.prices = prices
        self.features = features
        self.n_steps = len(prices) - 1  # Can trade for T-1 steps
        self.n_assets = prices.shape[1]
        self.initial_value = initial_value
        self.tc_rate = transaction_cost_rate
        self.slippage = slippage
        self.allow_short = allow_short
        self.max_position_size = max_position_size

        # State variables
        self.current_step = 0
        self.portfolio_value = initial_value
        self.current_weights = np.zeros(self.n_assets)  # Start with no positions
        self.portfolio_history = [initial_value]
        self.weight_history = [self.current_weights.copy()]
        self.return_history = []
        self.cost_history = []
        self.done = False

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.portfolio_value = self.initial_value
        self.current_weights = np.zeros(self.n_assets)
        self.portfolio_history = [self.initial_value]
        self.weight_history = [self.current_weights.copy()]
        self.return_history = []
        self.cost_history = []
        self.done = False

        return self._get_observation()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation (temporal context window)."""
        return {
            "features": self.features[self.current_step],     # (window_size, n_features)
            "current_weights": self.current_weights.copy(),     # (n_assets,)
            "portfolio_value": np.array([self.portfolio_value]),
            "step": self.current_step,
        }

    def _compute_transaction_costs(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
    ) -> float:
        """
        Compute transaction costs from rebalancing.
        
        Cost = tc_rate * |w_new - w_old| * portfolio_value
        """
        turnover = np.abs(new_weights - old_weights).sum()
        cost = self.tc_rate * turnover * self.portfolio_value
        return cost

    def _compute_slippage(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
    ) -> float:
        """Compute slippage cost from execution."""
        turnover = np.abs(new_weights - old_weights).sum()
        return self.slippage * turnover * self.portfolio_value

    def _clip_weights(self, weights: np.ndarray) -> np.ndarray:
        """Clip weights to respect position size limits."""
        if self.max_position_size < 1.0:
            weights = np.clip(weights, -self.max_position_size, self.max_position_size)
            # Re-normalize
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
        return weights

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: (n_assets,) target portfolio weights from Actor
            
        Returns:
            observation: next observation
            reward: step reward (to be computed by RewardCalculator)
            done: whether episode is finished
            info: additional information dict
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")

        # Clip weights
        new_weights = self._clip_weights(action.copy())

        # Compute costs
        tc_cost = self._compute_transaction_costs(self.current_weights, new_weights)
        slip_cost = self._compute_slippage(self.current_weights, new_weights)
        total_cost = tc_cost + slip_cost

        # Compute asset returns for this period
        current_prices = self.prices[self.current_step]
        next_prices = self.prices[self.current_step + 1]
        asset_returns = (next_prices - current_prices) / (current_prices + 1e-10)

        # Portfolio return (weighted sum of asset returns)
        portfolio_return = np.dot(new_weights, asset_returns)

        # Update portfolio value
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return) - total_cost

        # Update state
        self.current_weights = new_weights
        self.current_step += 1
        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(new_weights.copy())
        self.return_history.append(portfolio_return)
        self.cost_history.append(total_cost)

        # Check if done
        if self.current_step >= self.n_steps - 1:
            self.done = True
        if self.portfolio_value <= 0:
            self.done = True

        # Info dict
        info = {
            "portfolio_return": portfolio_return,
            "portfolio_value": self.portfolio_value,
            "transaction_cost": tc_cost,
            "slippage_cost": slip_cost,
            "total_cost": total_cost,
            "turnover": float(np.abs(new_weights - self.current_weights).sum()),
            "asset_returns": asset_returns,
        }

        observation = self._get_observation() if not self.done else None

        return observation, portfolio_return, self.done, info

    def get_performance_summary(self) -> Dict:
        """Get end-of-episode performance metrics."""
        portfolio_values = np.array(self.portfolio_history)
        returns = np.array(self.return_history) if self.return_history else np.array([0.0])

        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = (1 + total_return) ** (252 / max(len(returns), 1)) - 1
        annual_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        sharpe = annual_return / (annual_vol + 1e-10)

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / (peak + 1e-10)
        max_drawdown = drawdown.max()

        total_costs = sum(self.cost_history) if self.cost_history else 0.0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_transaction_costs": total_costs,
            "final_portfolio_value": portfolio_values[-1],
            "n_trading_days": len(returns),
        }
