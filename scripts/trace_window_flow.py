import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.config import get_config
from src.data.data_loader import CSEDataLoader
from src.data.preprocessor import Preprocessor
from src.environment.market_env import MarketEnvironment
from src.environment.reward import RewardCalculator
from src.models.ppo_agent import PPOAgent


def summarize_tensor(name, tensor, max_items=8):
    t = tensor.detach().cpu()
    if t.ndim == 0:
        return f"{name}: scalar={t.item():.6f}"
    if t.ndim == 1:
        return f"{name}: shape={tuple(t.shape)} values={t[:max_items].tolist()}"
    if t.ndim == 2:
        return f"{name}: shape={tuple(t.shape)} sample={t[0, :max_items].tolist()}"
    return f"{name}: shape={tuple(t.shape)} sample={t[0, 0, :max_items].tolist()}"


def main():
    device = torch.device("cpu")
    config = get_config(seed=42, device="cpu")
    config.data.n_assets = 10
    config.data.window_size = 30

    root = Path(__file__).resolve().parent.parent
    loader = CSEDataLoader(raw_data_dir=str(root / "data" / "raw"))
    prices, volumes = loader.load_prices_and_volumes(n_assets=config.data.n_assets)

    pre = Preprocessor(window_size=config.data.window_size, normalize_method="zscore")
    features_all = pre.engineer_features(prices, volumes, macro_data=None)
    features_all = pre.reorder_per_asset(features_all, list(prices.columns))
    norm_features = pre.fit_normalize(features_all)
    windows = pre.create_windows(norm_features.values)

    prices_aligned = prices.iloc[config.data.window_size - 1 : config.data.window_size - 1 + len(windows)].values

    n_steps = 5
    features_trace = windows[:n_steps]
    prices_trace = prices_aligned[: n_steps + 2]

    env = MarketEnvironment(
        prices=prices_trace,
        features=features_trace,
        initial_value=1_000_000.0,
        transaction_cost_rate=config.reward.transaction_cost_rate,
        slippage=config.environment.slippage,
        allow_short=config.environment.allow_short_selling,
        max_position_size=config.environment.max_position_size,
        vol_target=config.environment.vol_target,
        vol_ewma_span=config.environment.vol_ewma_span,
    )

    agent = PPOAgent(
        n_features_per_asset=pre.n_features_per_asset,
        n_assets=config.data.n_assets,
        n_macro=0,
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
        encoder_dropout=config.regime_encoder.lstm_dropout,
        mha_dropout=config.regime_encoder.mha_dropout,
        actor_dropout=config.actor.dropout,
        critic_dropout=config.critic.dropout,
        allow_short=config.environment.allow_short_selling,
    ).to(device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    reward_calc = RewardCalculator()

    print("=" * 90)
    print("TRACE: first 5 windows through the real PPO pipeline")
    print("=" * 90)
    print(f"Loaded {len(prices.columns)} assets from raw data")
    print(f"Window shape: {features_trace[0].shape}")
    print(f"Agent parameters: {agent.count_parameters()}")
    print()

    history = []
    obs = env.reset()

    for step_idx in range(n_steps):
        window_input = obs["features"]
        window_tensor = torch.tensor(window_input, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"Window {step_idx + 1}/{n_steps}")
        print(f"Input window shape: {tuple(window_tensor.shape)}")
        print(f"Input window sample (first 3 features of first time-step): {window_tensor[0, 0, :3].tolist()}")

        agent.train()
        model_out = agent(window_tensor, deterministic=False)

        action = model_out["action"][0].detach().cpu().numpy()
        raw_action = model_out["raw_action"][0].detach().cpu().numpy()
        log_prob = model_out["log_prob"][0]
        entropy = model_out["entropy"][0]
        value = model_out["value"][0]

        print("  Encoder")
        print(f"    h_asset: {tuple(model_out['h_asset'].shape)}")
        print(f"    h_market: {tuple(model_out['h_market'].shape)}")
        print(f"    regime_probs: {tuple(model_out['regime_probs'].shape)} -> {model_out['regime_probs'][0].cpu().tolist()}")
        print(f"    var_weights shape: {tuple(model_out['var_weights'].shape)}")
        print(f"    attn_weights shape: {tuple(model_out['attn_weights'].shape)}")

        print("  Actor")
        print(f"    raw_action: {raw_action[:4].tolist()}")
        print(f"    normalized weights: {action[:4].tolist()} ... sum={action.sum():.4f}")
        print(f"    log_prob: {log_prob.item():.6f}")
        print(f"    entropy: {entropy.item():.6f}")

        print("  Critic")
        print(f"    value estimate: {value.item():.6f}")

        old_weights = env.current_weights.copy()
        next_obs, portfolio_return, done, info = env.step(action)

        reward_info = reward_calc.compute_reward(portfolio_return, old_weights, action)
        reward_tensor = torch.tensor([reward_info["net_reward"]], dtype=torch.float32, device=device)
        advantage = reward_tensor - value.detach().unsqueeze(0)

        policy_loss = -(log_prob.unsqueeze(0) * advantage).mean()
        value_loss = F.mse_loss(value.unsqueeze(0), reward_tensor)
        entropy_loss = -entropy.unsqueeze(0).mean()
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        grad_norm = 0.0
        for p in agent.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        print("  Reward / environment")
        print(f"    portfolio_return: {portfolio_return:.6f}")
        print(f"    reward_components: {reward_info}")
        print(f"    total_loss: {total_loss.item():.6f}")
        print(f"    grad_norm: {grad_norm:.6f}")
        print()

        history.append(
            {
                "window_index": step_idx + 1,
                "reward": reward_info["net_reward"],
                "portfolio_return": portfolio_return,
                "regime_probs": model_out["regime_probs"][0].detach().cpu().tolist(),
                "weights": action.tolist(),
                "value": value.item(),
                "loss": total_loss.item(),
                "grad_norm": grad_norm,
            }
        )

        if done:
            break
        obs = next_obs

    out_path = root / "results" / "window_trace_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved detailed trace to {out_path}")


if __name__ == "__main__":
    main()
