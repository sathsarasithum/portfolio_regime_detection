"""
Central configuration for the PPO Portfolio Management System.
All hyperparameters, paths, and settings are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_MACRO_DIR = os.path.join(PROJECT_ROOT, "data", "macroeconomic")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(PROJECT_ROOT, "experiments", "logs")


@dataclass
class DataConfig:
    """Data & Preprocessing Configuration."""
    # Lookback window for temporal context x_t
    window_size: int = 25              # Adjusted to avoid ValueErrors with small data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Feature engineering
    use_returns: bool = True           # Log returns
    use_volatility: bool = True        # Rolling volatility
    use_momentum: bool = True          # Momentum indicators
    use_volume_features: bool = True   # Volume-based features
    normalize_method: str = "zscore"   # "zscore" | "minmax" | "robust"

    # Assets — will be populated from data
    asset_names: List[str] = field(default_factory=list)
    n_assets: int = 0
    n_macro_variables: int = 0


@dataclass
class RegimeEncoderConfig:
    """Differentiable Regime Encoder Configuration."""
    # V-VSN + LSTM
    vsn_hidden_dim: int = 64           # Variable Selection Network hidden dim
    lstm_hidden_dim: int = 128         # LSTM hidden dimension
    lstm_num_layers: int = 2           # Number of LSTM layers
    lstm_dropout: float = 0.1          # Dropout used within the regime encoder (VSN/LSTM, graph prior, fusion, regime classifier)

    # Temporal Attention (Multi-Head Attention)
    mha_num_heads: int = 4
    mha_dim: int = 128                 # Must match lstm_hidden_dim
    mha_dropout: float = 0.1

    # Macro Graph Prior (GNN/GAT)
    gnn_hidden_dim: int = 64
    gnn_num_heads: int = 2             # GAT attention heads
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.1

    # Latent regime representation
    n_regimes: int = 3                 # Bull, Bear, Sideways
    latent_state_dim: int = 128        # h_temp dimension


@dataclass
class ActorConfig:
    """Actor Network (Policy π) Configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"
    output_activation: str = "tanh"    # Portfolio weights via tanh
    dropout: float = 0.1


@dataclass
class CriticConfig:
    """Critic Network (Value V) Configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"
    dropout: float = 0.1


@dataclass
class PPOConfig:
    """PPO Training Configuration."""
    # Core PPO hyperparameters
    clip_epsilon: float = 0.2          # PPO clipping parameter
    gamma: float = 0.99                # Discount factor
    gae_lambda: float = 0.95          # GAE lambda
    #entropy_coeff: float = 0.01       # Entropy bonus coefficient
    entropy_coeff: float = 0.02
    value_loss_coeff: float = 0.5     # Value loss weight

    # Optimization
    #learning_rate: float = 3e-4
    learning_rate: float = 1e-4
    max_grad_norm: float = 0.5        # Gradient clipping
    batch_size: int = 64
    n_epochs: int = 10                 # PPO update epochs per rollout
    rollout_length: int = 256         # Steps per rollout

    # Training schedule
    total_timesteps: int = 500_000
    eval_frequency: int = 5_000       # Evaluate every N steps
    save_frequency: int = 10_000      # Save checkpoint every N steps
    early_stopping_patience: int = 20


@dataclass
class RewardConfig:
    """Reward Calculation Module Configuration."""
    # Net Reward = Sharpe Ratio − Costs + EVaR Penalty
    risk_free_rate: float = 0.06      # Sri Lanka ~6% annual
    transaction_cost_rate: float = 0.001  # 0.1% per trade (CSE)
    sharpe_window: int = 20           # Rolling window for Sharpe calculation
    evar_confidence: float = 0.95     # EVaR confidence level
    evar_penalty_weight: float = 0.1  # Weight for EVaR penalty
    turnover_penalty: float = 0.001   # Penalty for excessive turnover


@dataclass
class EnvironmentConfig:
    """Market Environment Configuration."""
    initial_portfolio_value: float = 1_000_000.0   # LKR 1M initial capital
    #allow_short_selling: bool = True                # Long/short — DeePM-style independent per-asset tanh signal.
                                                      # NOTE: CSE short-selling/SBL availability for your account
                                                      # type is unverified — confirm before treating results as
                                                      # executable. Set back to False to revert to long-only softmax.
    allow_short_selling: bool = False                # Long/short — DeePM-style independent per-asset tanh signal.                                                  
    max_position_size: float = 0.30                 # Max 30% in single asset
    rebalance_frequency: int = 1                    # Daily rebalancing
    slippage: float = 0.0005                        # 0.05% slippage
    vol_target: bool = True                         # Scale the long/short tanh signal by
                                                      # relative inverse volatility (DeePM Eq. 11)
                                                      # before position-size clipping. No effect
                                                      # when allow_short_selling is False.
    vol_ewma_span: int = 63                          # EWMA lookback (days) for the vol estimate


@dataclass
class ExperimentConfig:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    regime_encoder: RegimeEncoderConfig = field(default_factory=RegimeEncoderConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # Experiment metadata
    experiment_name: str = "cse_ppo_regime_v1"
    seed: int = 42
    device: str = "cuda"               # "cuda" | "cpu"
    num_workers: int = 4

    def __post_init__(self):
        """Create directories if they don't exist."""
        for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_MACRO_DIR,
                  MODELS_DIR, FIGURES_DIR, LOGS_DIR]:
            os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Default configuration instance
# ──────────────────────────────────────────────────────────────
def get_config(**overrides) -> ExperimentConfig:
    """Get experiment configuration with optional overrides."""
    config = ExperimentConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
