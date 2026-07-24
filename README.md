# Jointly Optimized, End-to-End PPO Reinforcement Learning for Quantitative Portfolio Management

## Research Overview

This project implements a **Jointly Optimized, End-to-End PPO-based Portfolio Architecture** that integrates:

1. **Differentiable Regime Encoder** — V-VSN + LSTM with Temporal Attention (MHA) and a Macro Graph Prior (GNN/GAT) to produce latent market states and detect regimes (Bull / Bear / Sideways) in real-time.
2. **Actor-Critic PPO Agent** — Policy network (Linear+tanh) outputs portfolio weights; Critic network (Linear+scalar) estimates state value.
3. **Custom Reward Module** — Net Reward = Sharpe Ratio − Transaction Costs + EVaR Penalty, with Advantage Estimation for stable policy gradients.
4. **Joint End-to-End Gradient Update** — All components (encoder, actor, critic) are trained jointly via backpropagation.

**Data**: 3 years of Colombo Stock Exchange (CSE) market data (prices, volumes, macroeconomic indicators).

---

## Project Structure

```
Regime detection portfolio/
│
├── data/
│   ├── raw/                        # Raw CSE data files (upload here)
│   ├── processed/                  # Cleaned & feature-engineered data
│   └── macroeconomic/              # Macroeconomic indicator data
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py               # All hyperparameters & settings
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Load raw CSE data
│   │   ├── preprocessor.py         # Windowing, feature engineering
│   │   └── dataset.py              # PyTorch Dataset & DataLoader
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vsn_lstm.py             # Variable Selection Network + LSTM
│   │   ├── temporal_attention.py   # Multi-Head Attention module
│   │   ├── macro_graph_prior.py    # GNN/GAT for macroeconomic graph
│   │   ├── regime_encoder.py       # Full Differentiable Regime Encoder
│   │   ├── actor.py                # Actor Network (Policy π)
│   │   ├── critic.py               # Critic Network (Value V)
│   │   └── ppo_agent.py            # Combined PPO Agent
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── market_env.py           # Market Environment (execute trades)
│   │   └── reward.py               # Reward Calculation Module
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # PPO training loop
│   │   ├── advantage.py            # GAE (Generalized Advantage Estimation)
│   │   └── optimizer.py            # Joint end-to-end optimizer setup
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py              # Sharpe, EVaR, drawdown, etc.
│       ├── visualization.py        # Plotting & analysis
│       └── logger.py               # Experiment logging
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Data preprocessing walkthrough
│   ├── 03_training.ipynb           # Training & experimentation
│   └── 04_evaluation.ipynb         # Results & evaluation
│
├── experiments/
│   └── logs/                       # TensorBoard / W&B logs
│
├── results/
│   ├── figures/                    # Generated plots & charts
│   ├── models/                     # Saved model checkpoints
│   └── reports/                    # Performance reports
│
├── tests/
│   ├── __init__.py
│   └── test_env.py                 # Unit tests
│
├── requirements.txt                # Python dependencies
├── main.py                         # Entry point: train / evaluate
└── README.md                       # This file
```

---

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train the model
python main.py --mode train

# Evaluate
python main.py --mode evaluate

# Backtest
python main.py --mode backtest




#What the Critic Network does
#In this model, the critic network estimates the value of the current market state.

#Specifically
#It receives:

        #h_temp: the latent market state produced by the regime encoder
        #regime_probs: the soft regime probability vector

#It outputs:
#       a scalar value estimate V(s) for the current state#

#Why it matters
#PPO uses the critic to compute the advantage:
        #advantage = actual_return - value_estimate
#That advantage tells the actor how much better or worse the action was than expected.
#The critic is also trained with a value loss, so it learns to predict future reward more accurately over time.

#So the critic is not choosing portfolio weights. It is the value estimator that guides policy updates and stabilizes PPO training.
```
