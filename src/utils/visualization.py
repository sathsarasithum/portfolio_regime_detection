"""
Visualization Module
=====================
Plotting and analysis utilities for portfolio performance,
regime detection, and attention visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_portfolio_performance(
    portfolio_values: np.ndarray,
    benchmark_values: Optional[np.ndarray] = None,
    title: str = "Portfolio Performance",
    save_path: Optional[str] = None,
):
    """Plot portfolio value over time with optional benchmark comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

    # Portfolio value
    axes[0].plot(portfolio_values, label="PPO Portfolio", linewidth=2, color="#2196F3")
    if benchmark_values is not None:
        axes[0].plot(benchmark_values, label="Benchmark", linewidth=1.5,
                     color="#FF9800", alpha=0.7)
    axes[0].set_title(title, fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Portfolio Value (LKR)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / (peak + 1e-10)
    axes[1].fill_between(range(len(drawdown)), drawdown, alpha=0.4, color="#F44336")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Trading Day")
    axes[1].set_ylim(0, drawdown.max() * 1.1)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot: {save_path}")
    plt.close()


def plot_regime_detection(
    regime_probs: np.ndarray,
    portfolio_values: np.ndarray,
    regime_names: List[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot detected market regimes overlaid on portfolio value.
    
    Args:
        regime_probs: (T, n_regimes) regime probabilities
        portfolio_values: (T,) portfolio values
    """
    if regime_names is None:
        regime_names = ["Bull", "Bear", "Sideways"]

    colors = ["#4CAF50", "#F44336", "#FF9800"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    # Portfolio with regime coloring
    dominant_regime = regime_probs.argmax(axis=1)
    for i, (name, color) in enumerate(zip(regime_names, colors)):
        mask = dominant_regime == i
        if mask.any():
            indices = np.where(mask)[0]
            for idx in indices:
                axes[0].axvspan(idx - 0.5, idx + 0.5, alpha=0.15, color=color)

    axes[0].plot(portfolio_values, color="black", linewidth=1.5)
    axes[0].set_title("Market Regime Detection", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Portfolio Value")

    # Add legend patches
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=c, alpha=0.3, label=n)
        for n, c in zip(regime_names, colors)
    ]
    axes[0].legend(handles=legend_patches, loc="upper left")

    # Regime probabilities
    for i, (name, color) in enumerate(zip(regime_names, colors)):
        axes[1].plot(regime_probs[:, i], label=name, color=color, linewidth=1.5)
    axes[1].set_ylabel("Regime Probability")
    axes[1].set_xlabel("Trading Day")
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_weight_allocation(
    weights: np.ndarray,
    asset_names: List[str] = None,
    save_path: Optional[str] = None,
):
    """Plot portfolio weight allocation over time."""
    n_assets = weights.shape[1]
    if asset_names is None:
        asset_names = [f"Asset {i+1}" for i in range(n_assets)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(
        range(len(weights)),
        *[weights[:, i] for i in range(n_assets)],
        labels=asset_names,
        alpha=0.8,
    )
    ax.set_title("Portfolio Weight Allocation Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", ncol=min(5, n_assets))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    title: str = "Temporal Attention Weights",
    save_path: Optional[str] = None,
):
    """Plot attention heatmap for temporal attention visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        cmap="YlOrRd",
        ax=ax,
        xticklabels=10,
        yticklabels=10,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_variable_importance(
    var_weights: np.ndarray,
    feature_names: List[str] = None,
    top_k: int = 20,
    save_path: Optional[str] = None,
):
    """Plot variable importance from VSN."""
    avg_importance = var_weights.mean(axis=(0, 1))  # Average over batch & time

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(avg_importance))]

    # Top-K features
    top_idx = np.argsort(avg_importance)[-top_k:]
    top_names = [feature_names[i] for i in top_idx]
    top_values = avg_importance[top_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    bars = ax.barh(top_names, top_values, color="#2196F3", alpha=0.8)
    ax.set_title(f"Top {top_k} Variable Importance (VSN)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Importance Weight")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history: List[Dict],
    save_path: Optional[str] = None,
):
    """Plot training loss and reward curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = [h["step"] for h in history]

    # Total loss
    axes[0, 0].plot(steps, [h["total_loss"] for h in history], color="#2196F3")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Step")

    # Policy loss
    axes[0, 1].plot(steps, [h["policy_loss"] for h in history], color="#4CAF50")
    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_xlabel("Step")

    # Value loss
    axes[1, 0].plot(steps, [h["value_loss"] for h in history], color="#FF9800")
    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Step")

    # Average reward
    axes[1, 1].plot(steps, [h["avg_reward"] for h in history], color="#9C27B0")
    axes[1, 1].set_title("Average Reward")
    axes[1, 1].set_xlabel("Step")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_summary(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
):
    """Create a visual summary of portfolio metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    keys = list(metrics.keys())
    values = list(metrics.values())

    colors = ["#4CAF50" if v > 0 else "#F44336" for v in values]
    bars = ax.barh(keys, values, color=colors, alpha=0.8)

    ax.set_title("Portfolio Performance Metrics", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
