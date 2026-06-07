"""
CSE PPO Portfolio Management System - Entrypoint
=================================================
Coordinates data loading, preprocessing, model initialization, and training/evaluation.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import logging

from src.config.config import get_config, DATA_RAW_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR
from src.data.data_loader import CSEDataLoader
from src.data.preprocessor import Preprocessor
from src.environment.market_env import MarketEnvironment
from src.models.ppo_agent import PPOAgent
from src.training.trainer import PPOTrainer
from src.utils.logger import setup_logger

logger = logging.getLogger("portfolio_rl")


def parse_args():
    parser = argparse.ArgumentParser(description="Jointly Optimized PPO Portfolio Management System")
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "eval", "backtest"],
        help="Execution mode: train, eval, or backtest"
    )
    parser.add_argument(
        "--output-dir", type=str, default=RESULTS_DIR,
        help="Directory to save evaluation summaries and results"
    )
    parser.add_argument(
        "--n-assets", type=int, default=47,
        help="Number of assets (companies) to select from raw data. Set to 47 for 0% missing values."
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=10000,
        help="Total PPO training timesteps"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="PPO training epochs per rollout"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="PPO optimization batch size"
    )
    parser.add_argument(
        "--rollout-length", type=int, default=256,
        help="PPO rollout buffer length"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Optimizer learning rate"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best_model.pt",
        help="Checkpoint filename for evaluation or resuming"
    )
    parser.add_argument(
        "--eval-frequency", type=int, default=5000,
        help="Timesteps frequency for evaluation"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=10000,
        help="Timesteps frequency for saving checkpoints"
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(args):
    # Setup central logger
    setup_logger(name="portfolio_rl", log_dir=LOGS_DIR)
    logger.info("=" * 60)
    logger.info("Initializing Jointly Optimized PPO Portfolio Management System")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")

    set_seed(args.seed)

    # 1. Load configuration overrides
    config = get_config(
        seed=args.seed,
        device=args.device,
    )
    config.ppo.total_timesteps = args.total_timesteps
    config.ppo.n_epochs = args.epochs
    config.ppo.batch_size = args.batch_size
    config.ppo.rollout_length = args.rollout_length
    config.ppo.learning_rate = args.lr
    config.ppo.eval_frequency = args.eval_frequency
    config.ppo.save_frequency = args.save_frequency
    config.data.n_assets = args.n_assets
    config.data.n_macro_variables = 0  # CRITICAL: Macroeconomic variables disabled for this phase

    # 2. Load Prices and Volumes via CSEDataLoader
    logger.info(f"Loading raw CSE data from {DATA_RAW_DIR}...")
    loader = CSEDataLoader(raw_data_dir=DATA_RAW_DIR)
    prices, volumes = loader.load_prices_and_volumes(n_assets=config.data.n_assets)

    config.data.asset_names = list(prices.columns)
    config.data.n_assets = len(config.data.asset_names)
    logger.info(f"Successfully loaded and aligned {config.data.n_assets} assets.")

    # 3. Preprocess Data and Engineer Features
    preprocessor = Preprocessor(
        window_size=config.data.window_size,
        normalize_method=config.data.normalize_method
    )
    logger.info("Engineering features across the full dataset...")
    features_all = preprocessor.engineer_features(prices, volumes, macro_data=None)

    # Align prices to engineered features
    prices = prices.loc[features_all.index]
    volumes = volumes.loc[features_all.index]

    # 4. Perform Chronological Splits (Train / Val / Test)
    n_total = len(features_all)
    n_train = int(n_total * config.data.train_ratio)
    n_val = int(n_total * config.data.val_ratio)

    features_train = features_all.iloc[:n_train]
    features_val = features_all.iloc[n_train:n_train + n_val]
    features_test = features_all.iloc[n_train + n_val:]

    prices_train = prices.loc[features_train.index]
    prices_val = prices.loc[features_val.index]
    prices_test = prices.loc[features_test.index]

    logger.info(f"Splits - Train: {len(features_train)} steps, Val: {len(features_val)} steps, Test: {len(features_test)} steps")

    # 5. Fit & Apply Normalization (No Data Leakage)
    norm_train = preprocessor.fit_normalize(features_train)
    norm_val = preprocessor.transform_normalize(features_val)
    norm_test = preprocessor.transform_normalize(features_test)

    # 6. Create Temporal Sliding Observation Windows (x_t)
    windows_train = preprocessor.create_windows(norm_train.values)
    windows_val = preprocessor.create_windows(norm_val.values)
    windows_test = preprocessor.create_windows(norm_test.values)

    # Align prices to the window end dates
    prices_train_aligned = prices_train.iloc[config.data.window_size - 1:].values
    prices_val_aligned = prices_val.iloc[config.data.window_size - 1:].values
    prices_test_aligned = prices_test.iloc[config.data.window_size - 1:].values

    logger.info(f"Windows created - Train: {windows_train.shape[0]}, Val: {windows_val.shape[0]}, Test: {windows_test.shape[0]}")

    # 7. Build Market Environments
    train_env = MarketEnvironment(
        prices=prices_train_aligned,
        features=windows_train,
        initial_value=config.environment.initial_portfolio_value,
        transaction_cost_rate=config.reward.transaction_cost_rate,
        slippage=config.environment.slippage,
        allow_short=config.environment.allow_short_selling,
        max_position_size=config.environment.max_position_size,
    )

    val_env = MarketEnvironment(
        prices=prices_val_aligned,
        features=windows_val,
        initial_value=config.environment.initial_portfolio_value,
        transaction_cost_rate=config.reward.transaction_cost_rate,
        slippage=config.environment.slippage,
        allow_short=config.environment.allow_short_selling,
        max_position_size=config.environment.max_position_size,
    )

    test_env = MarketEnvironment(
        prices=prices_test_aligned,
        features=windows_test,
        initial_value=config.environment.initial_portfolio_value,
        transaction_cost_rate=config.reward.transaction_cost_rate,
        slippage=config.environment.slippage,
        allow_short=config.environment.allow_short_selling,
        max_position_size=config.environment.max_position_size,
    )

    # 8. Initialize joint PPO Agent
    n_features = windows_train.shape[2]  # typically config.data.n_assets * 15
    logger.info(f"Initializing PPO Agent with {config.data.n_assets} assets and {n_features} features...")
    agent = PPOAgent(
        n_features=n_features,
        n_assets=config.data.n_assets,
        n_macro=0,  # CRITICAL: No macroeconomic variables
        vsn_hidden_dim=config.regime_encoder.vsn_hidden_dim,
        lstm_hidden_dim=config.regime_encoder.lstm_hidden_dim,
        lstm_num_layers=config.regime_encoder.lstm_num_layers,
        mha_num_heads=config.regime_encoder.mha_num_heads,
        gnn_hidden_dim=config.regime_encoder.gnn_hidden_dim,
        gnn_num_heads=config.regime_encoder.gnn_num_heads,
        gnn_num_layers=config.regime_encoder.gnn_num_layers,
        n_regimes=config.regime_encoder.n_regimes,
        latent_state_dim=config.regime_encoder.latent_state_dim,
        actor_hidden_dims=config.actor.hidden_dims,
        critic_hidden_dims=config.critic.hidden_dims,
        dropout=config.regime_encoder.lstm_dropout,
        allow_short=config.environment.allow_short_selling,
    )

    # 9. Execute Mode
    if args.mode == "train":
        logger.info("Initializing PPOTrainer...")
        trainer = PPOTrainer(
            agent=agent,
            train_env=train_env,
            val_env=val_env,
            clip_epsilon=config.ppo.clip_epsilon,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            entropy_coeff=config.ppo.entropy_coeff,
            value_loss_coeff=config.ppo.value_loss_coeff,
            learning_rate=config.ppo.learning_rate,
            max_grad_norm=config.ppo.max_grad_norm,
            batch_size=config.ppo.batch_size,
            n_epochs=config.ppo.n_epochs,
            rollout_length=config.ppo.rollout_length,
            total_timesteps=config.ppo.total_timesteps,
            eval_frequency=config.ppo.eval_frequency,
            save_frequency=config.ppo.save_frequency,
            early_stopping_patience=config.ppo.early_stopping_patience,
            log_dir=LOGS_DIR,
            save_dir=MODELS_DIR,
            device=config.device,
        )
        logger.info("Starting training loop...")
        trainer.train()
        logger.info("Training finished. Evaluating on test set...")

        # Load best model for test evaluation
        best_model_path = os.path.join(MODELS_DIR, "best_model.pt")
        if os.path.exists(best_model_path):
            trainer.load_checkpoint(best_model_path)
            logger.info("Loaded best model checkpoint for testing.")

        # Test evaluation
        evaluate_env(agent, test_env, config.device, args.output_dir, "Test")

    elif args.mode == "eval":
        checkpoint_path = os.path.join(MODELS_DIR, args.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint {checkpoint_path} for evaluation...")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        agent.to(args.device)

        # Evaluation on test set
        evaluate_env(agent, test_env, args.device, args.output_dir, "Test")
    elif args.mode == "backtest":
        checkpoint_path = os.path.join(MODELS_DIR, args.checkpoint)
        if not os.path.exists(checkpoint_path):
            best_model_path = os.path.join(MODELS_DIR, "best_model.pt")
            if os.path.exists(best_model_path):
                checkpoint_path = best_model_path
                logger.info(f"No checkpoint specified. Using best model: {checkpoint_path}")
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {args.checkpoint} and no best_model.pt in {MODELS_DIR}"
                )

        logger.info(f"Loading checkpoint {checkpoint_path} for backtest...")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        agent.to(args.device)

        evaluate_env(agent, test_env, args.device, args.output_dir, "Backtest")


def evaluate_env(agent, env, device, output_dir, split_name="Test"):
    agent.eval()
    obs = env.reset()
    lstm_hidden = None
    done = False

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{split_name.lower()}_summary.json")

    logger.info(f"Running simulation on {split_name} environment...")
    while not done:
        obs_tensor = torch.FloatTensor(obs["features"]).unsqueeze(0).to(device)
        with torch.no_grad():
            action, info = agent.get_action(obs_tensor, lstm_hidden=lstm_hidden, deterministic=True)
            lstm_hidden = info["lstm_hidden"]

        action_np = action.cpu().numpy()[0]
        next_obs, _, done, _ = env.step(action_np)
        if next_obs is None:
            break
        obs = next_obs

    summary = env.get_performance_summary()
    logger.info("=" * 60)
    logger.info(f"{split_name} Performance Summary:")
    logger.info("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Saved evaluation summary to: {summary_path}")
    except Exception as e:
        logger.warning(f"Failed to save evaluation summary: {e}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
