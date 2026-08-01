"""
Market Environment
===================
Simulates the trading environment for portfolio management.
Executes trades, tracks portfolio value, and calculates transaction costs.
"""

import numpy as np
import pandas as pd
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
        vol_target: bool = True,
        vol_ewma_span: int = 63,
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
            vol_target: if True (and allow_short), scale the actor's raw
                tanh signal by each asset's inverse volatility before
                clipping — see _compute_vol_scale() for why and how.
            vol_ewma_span: EWMA span (in days) for the volatility estimate,
                63 days (~1 quarter) following DeePM (Eq. 11).
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
        self.vol_target = vol_target
        self.vol_ewma_span = vol_ewma_span
        self.vol_scale = self._compute_vol_scale() if vol_target else None

        # State variables
        self.current_step = 0
        self.portfolio_value = initial_value
        self.current_weights = np.zeros(self.n_assets)  # Start with no positions
        self.portfolio_history = [initial_value]
        self.weight_history = [self.current_weights.copy()]
        self.return_history = []
        self.cost_history = []
        self.done = False

    def _compute_vol_scale(self) -> np.ndarray:
        """
        Per-asset, per-step relative volatility scaling factor for the
        long/short tanh action space (DeePM Eq. 11: v = 1/sigma_hat).

        Why this lives here, not in the actor: the actor only ever sees
        z-scored engineered features, not real prices, so it has no basis
        for comparing assets' actual risk. This project's own 10 banks span
        a genuine ~10x range in daily volatility (e.g. HNB ~1% vs NTB
        ~10%), so an identical tanh signal on two assets currently implies
        wildly different real risk — this is what that scaling corrects.

        Why "relative" (mean / sigma_i) rather than a raw 1/sigma_i
        multiplier: a raw 1/sigma_i is not scale-free — for a low-vol asset
        like HNB (sigma~0.0107) it's ~93x, which would saturate
        max_position_size for almost any nonzero signal, while DeePM avoids
        this because they rescale the *whole portfolio* to a target
        volatility ex-post (Sec 6.3) rather than capping each position
        individually. Centering the scale factor at 1.0 across the
        cross-section keeps "an average-risk asset" behaving like the
        unscaled model did, while shifting relative sizing toward safer
        names — compatible with the existing fixed max_position_size cap.

        No-lookahead note: sigma_hat for the decision made at step t uses
        only returns realized up to (not including) step t, i.e. it is
        shifted by one day relative to the raw EWMA. Day 0 is the one
        exception — it reuses day 0's own trailing estimate since no prior
        return exists yet; this is a one-day, first-step-only approximation.

        Returns:
            vol_scale: (n_steps, n_assets) multiplier, apply as
                `action * vol_scale[current_step]` BEFORE _clip_weights.
        """
        log_returns = np.diff(np.log(self.prices + 1e-10), axis=0)  # (n_steps, n_assets)
        ewma_vol = (
            pd.DataFrame(log_returns)
            .ewm(span=self.vol_ewma_span, min_periods=1)
            .std()
            .bfill()
            .values
        )  # (n_steps, n_assets), row i estimated from returns[0..i]

        # Shift by one day: the estimate usable at decision time t must not
        # include day t's own return.
        sigma_hat = np.vstack([ewma_vol[0:1], ewma_vol[:-1]])
        sigma_hat = np.maximum(sigma_hat, 1e-6)  # guard near-zero vol

        cross_sectional_mean = sigma_hat.mean(axis=1, keepdims=True)
        return cross_sectional_mean / sigma_hat  # (n_steps, n_assets), centered ~1.0

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
        """
        Clip weights to respect position size limits.

        Long-only: cap each weight at max_position_size, then renormalize
        the capped weights back onto the simplex (sum to 1, fully invested).

        Long-short: cap each weight's MAGNITUDE at max_position_size and
        leave it there — no renormalization. Dividing by weights.sum() is
        wrong here: with signed weights the sum is the NET exposure, which
        can be small or negative even when positions are large (e.g.
        weights=[+0.3,-0.28,...] sums to ~0.02), so dividing by it explodes
        the position sizes arbitrarily (verified: a sum of 0.0248 inflated a
        single weight to 7.75, i.e. 775% of capital in one asset). Relative
        volatility scaling (see _compute_vol_scale, applied in step() before
        this is called) reduces how often the cap binds for high-vol assets,
        but max_position_size remains the hard exposure control: gross
        exposure is bounded by n_assets * max_position_size.
        """
        if self.max_position_size < 1.0:
            weights = np.clip(weights, -self.max_position_size, self.max_position_size)
            if not self.allow_short:
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

        # Volatility-scaled notional sizing (DeePM Eq. 11-12): rescale the
        # actor's raw per-asset signal by relative inverse volatility before
        # applying position-size limits. Only meaningful for the
        # independent long/short signal — the long-only softmax weights are
        # already a resource-constrained allocation, not a signal to scale.
        if self.allow_short and self.vol_target:
            action = action * self.vol_scale[self.current_step]

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
