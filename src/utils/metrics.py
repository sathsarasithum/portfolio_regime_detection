"""
Portfolio Performance Metrics
==============================
Sharpe ratio, EVaR, maximum drawdown, Calmar ratio, and other
portfolio performance metrics.
"""

import numpy as np
from typing import Dict, Optional


class PortfolioMetrics:
    """Compute comprehensive portfolio performance metrics."""

    def __init__(self, risk_free_rate: float = 0.06):
        self.rf = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    def sharpe_ratio(self, returns: np.ndarray) -> float:
        """Annualized Sharpe Ratio."""
        excess = returns - self.daily_rf
        if excess.std() == 0:
            return 0.0
        return excess.mean() / excess.std() * np.sqrt(252)

    def sortino_ratio(self, returns: np.ndarray) -> float:
        """Annualized Sortino Ratio (downside deviation)."""
        excess = returns - self.daily_rf
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return excess.mean() / downside.std() * np.sqrt(252)

    def max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Maximum drawdown."""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / (peak + 1e-10)
        return drawdown.max()

    def calmar_ratio(self, returns: np.ndarray,
                     portfolio_values: np.ndarray) -> float:
        """Calmar Ratio = Annual Return / Max Drawdown."""
        ann_return = self.annualized_return(returns)
        mdd = self.max_drawdown(portfolio_values)
        if mdd == 0:
            return 0.0
        return ann_return / mdd

    def annualized_return(self, returns: np.ndarray) -> float:
        """Annualized return."""
        cumulative = (1 + returns).prod()
        n_years = len(returns) / 252
        if n_years == 0:
            return 0.0
        return cumulative ** (1/n_years) - 1

    def annualized_volatility(self, returns: np.ndarray) -> float:
        """Annualized volatility."""
        return returns.std() * np.sqrt(252)

    def value_at_risk(self, returns: np.ndarray,
                      confidence: float = 0.95) -> float:
        """Value at Risk (VaR) at given confidence level."""
        return np.percentile(returns, (1 - confidence) * 100)

    def conditional_var(self, returns: np.ndarray,
                        confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall / CVaR)."""
        var = self.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    def evar(self, returns: np.ndarray,
             confidence: float = 0.95) -> float:
        """
        Entropic Value-at-Risk (EVaR).
        Upper bound on both VaR and CVaR.
        """
        alpha = 1 - confidence
        z_values = np.linspace(0.01, 10, 1000)
        evar_values = []

        for z in z_values:
            exp_term = np.mean(np.exp(-z * returns))
            evar_val = (1/z) * np.log(exp_term / alpha)
            evar_values.append(evar_val)

        return min(evar_values)

    def information_ratio(self, returns: np.ndarray,
                          benchmark_returns: np.ndarray) -> float:
        """Information Ratio relative to benchmark."""
        active_returns = returns - benchmark_returns
        if active_returns.std() == 0:
            return 0.0
        return active_returns.mean() / active_returns.std() * np.sqrt(252)

    def win_rate(self, returns: np.ndarray) -> float:
        """Percentage of positive return days."""
        return (returns > 0).mean()

    def profit_factor(self, returns: np.ndarray) -> float:
        """Ratio of gross profits to gross losses."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float('inf')
        return gains / losses

    def compute_all(
        self,
        returns: np.ndarray,
        portfolio_values: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {
            "sharpe_ratio": self.sharpe_ratio(returns),
            "sortino_ratio": self.sortino_ratio(returns),
            "annualized_return": self.annualized_return(returns),
            "annualized_volatility": self.annualized_volatility(returns),
            "max_drawdown": self.max_drawdown(portfolio_values),
            "calmar_ratio": self.calmar_ratio(returns, portfolio_values),
            "var_95": self.value_at_risk(returns, 0.95),
            "cvar_95": self.conditional_var(returns, 0.95),
            "evar_95": self.evar(returns, 0.95),
            "win_rate": self.win_rate(returns),
            "profit_factor": self.profit_factor(returns),
            "total_return": (portfolio_values[-1] / portfolio_values[0]) - 1,
            "final_value": portfolio_values[-1],
        }

        if benchmark_returns is not None:
            metrics["information_ratio"] = self.information_ratio(
                returns, benchmark_returns
            )

        return metrics
