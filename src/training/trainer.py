"""
PPO Trainer
============
Main training loop implementing the Jointly Optimized, End-to-End
PPO algorithm for portfolio management.

Training Loop:
    1. Collect rollouts using current policy in MarketEnvironment
    2. Compute rewards using RewardCalculator
    3. Compute advantages using GAE
    4. Update all parameters jointly via PPO clipped objective
    5. Evaluate and log metrics
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
import logging

from ..models.ppo_agent import PPOAgent
from ..environment.market_env import MarketEnvironment
from ..environment.reward import RewardCalculator
from ..data.dataset import RolloutBuffer
from .advantage import compute_gae, normalize_advantages, RewardNormalizer
from .optimizer import create_optimizer, create_scheduler

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO Trainer with Joint End-to-End Optimization.
    
    Key: Gradients flow from the reward through the actor's weight
    predictions, back through the regime encoder. All components
    (VSN+LSTM, MHA, GNN/GAT, Actor, Critic) are updated jointly.
    """

    def __init__(
        self,
        agent: PPOAgent,
        train_env: MarketEnvironment,
        val_env: Optional[MarketEnvironment] = None,
        # PPO hyperparameters
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        # Optimization
        learning_rate: float = 3e-4,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        n_epochs: int = 10,
        rollout_length: int = 256,
        # Training schedule
        total_timesteps: int = 500_000,
        eval_frequency: int = 5000,
        save_frequency: int = 10000,
        early_stopping_patience: int = 20,
        # Paths
        log_dir: str = "experiments/logs",
        save_dir: str = "results/models",
        device: str = "cpu",
    ):
        self.agent = agent.to(device)
        self.train_env = train_env
        self.val_env = val_env
        self.device = device

        # PPO hyperparameters
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.rollout_length = rollout_length
        self.total_timesteps = total_timesteps
        self.eval_frequency = eval_frequency
        self.save_frequency = save_frequency
        self.early_stopping_patience = early_stopping_patience

        # Optimizer & scheduler
        self.optimizer = create_optimizer(agent, learning_rate)
        total_updates = total_timesteps // rollout_length * n_epochs
        self.scheduler = create_scheduler(self.optimizer, total_updates)

        # Reward calculator
        self.reward_calc = RewardCalculator()
        # Keeps discounted returns O(1) so the critic's gradient does not
        # dominate the shared global grad-norm clip and starve the actor.
        self.reward_normalizer = RewardNormalizer(gamma=gamma)

        # Logging
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.global_step = 0
        self.best_sharpe = -np.inf
        self.patience_counter = 0

    def collect_rollout(self) -> Dict:
        """
        Collect a rollout of transitions using the current policy.
        
        Returns:
            rollout dict with observations, actions, rewards, etc.
        """
        self.agent.eval()
        self.reward_calc.reset()

        observations = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        regime_probs_list = []
        portfolio_returns = []

        obs = self.train_env.reset()

        for step in range(self.rollout_length):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(
                obs["features"]
            ).unsqueeze(0).to(self.device)

            # Get action from policy.
            # No LSTM state is carried across steps: each observation is a
            # complete sliding window, and evaluate_actions() re-encodes with a
            # fresh hidden state. Carrying it here would make the stored
            # log_prob refer to a different state than the one re-evaluated
            # during the update, corrupting the PPO ratio from epoch 0.
            with torch.no_grad():
                out = self.agent(obs_tensor)

            weights = out["action"].cpu().numpy()[0]
            raw_action = out["raw_action"].cpu().numpy()[0]
            log_prob = out["log_prob"].cpu().item()
            value = out["value"].cpu().item()
            regime_probs = out["regime_probs"].cpu().numpy()[0]

            logger.info(
                f"Train step {step}: raw_action={raw_action.tolist()} normalized_weight={weights.tolist()}"
            )

            # Store old weights before step
            old_weights = self.train_env.current_weights.copy()

            # Step environment with the normalized allocation
            next_obs, portfolio_return, done, info = self.train_env.step(weights)

            # Compute reward
            reward_info = self.reward_calc.compute_reward(
                portfolio_return, old_weights, weights
            )
            reward = reward_info["net_reward"]

            # Store transition — the RAW action, since that is what log_prob
            # was computed for and what the update must re-evaluate.
            observations.append(obs["features"])
            actions.append(raw_action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(float(done))
            regime_probs_list.append(regime_probs)
            portfolio_returns.append(portfolio_return)

            if done:
                obs = self.train_env.reset()
                self.reward_calc.reset()
            else:
                obs = next_obs

        # Get value of last state for GAE
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(
                obs["features"]
            ).unsqueeze(0).to(self.device)
            last_out = self.agent(obs_tensor)
            last_value = last_out["value"].cpu().item()

        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "log_probs": np.array(log_probs),
            "values": np.array(values),
            "rewards": np.array(rewards),
            "portfolio_returns": np.array(portfolio_returns),
            "dones": np.array(dones),
            "regime_probs": np.array(regime_probs_list),
            "last_value": last_value,
        }

    def compute_ppo_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO clipped objective loss.
        
        L = L_clip + c1 * L_value + c2 * L_entropy
        
        Where:
            L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
            L_value = (V(s) - R_t)^2
            L_entropy = -H(π)
        """
        obs = batch["observations"]
        old_actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        # Evaluate actions under current policy
        eval_out = self.agent.evaluate_actions(obs, old_actions)
        new_log_probs = eval_out["log_prob"]
        values = eval_out["value"]
        entropy = eval_out["entropy"]

        # ── Policy Loss (Clipped PPO) ──
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value Loss ──
        value_loss = F.mse_loss(values, returns)

        # ── Entropy Loss ──
        entropy_loss = -entropy.mean()

        # ── Total Loss ──
        total_loss = (
            policy_loss
            + self.value_loss_coeff * value_loss
            + self.entropy_coeff * entropy_loss
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "approx_kl": ((ratio - 1) - ratio.log()).mean(),
            "clip_fraction": (
                (ratio - 1.0).abs() > self.clip_epsilon
            ).float().mean(),
        }

    def update(self, rollout: Dict, extra_epochs: int = 0) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout.
        Joint end-to-end gradient update through all components.
        """
        self.agent.train()

        # Convert to tensors
        observations = torch.FloatTensor(rollout["observations"]).to(self.device)
        actions = torch.FloatTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        values_np = rollout["values"]
        dones_np = rollout["dones"]

        # Scale rewards to unit discounted-return std before GAE, so that
        # rewards, values and returns all live in one consistent scale.
        # collect_rollout() always begins from env.reset(), so the discounted
        # -return accumulator restarts with it.
        self.reward_normalizer.reset()
        rewards_np = self.reward_normalizer(rollout["rewards"], dones_np)

        # Compute GAE advantages
        rewards_t = torch.FloatTensor(rewards_np).to(self.device)
        values_t = torch.FloatTensor(values_np).to(self.device)
        dones_t = torch.FloatTensor(dones_np).to(self.device)

        advantages, returns = compute_gae(
            rewards_t, values_t, dones_t,
            torch.tensor(rollout["last_value"]).to(self.device),
            self.gamma, self.gae_lambda,
        )
        advantages = normalize_advantages(advantages)

        # PPO update epochs
        epoch_losses = []
        for epoch in range(self.n_epochs + extra_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(observations))
            for start in range(0, len(observations), self.batch_size):
                end = min(start + self.batch_size, len(observations))
                idx = indices[start:end]

                batch = {
                    "observations": observations[idx],
                    "actions": actions[idx],
                    "log_probs": old_log_probs[idx],
                    "returns": returns[idx],
                    "advantages": advantages[idx],
                }

                # Compute loss
                losses = self.compute_ppo_loss(batch)

                # Joint end-to-end gradient update
                self.optimizer.zero_grad()
                losses["total_loss"].backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()

                epoch_losses.append({
                    k: v.item() for k, v in losses.items()
                })

        # Average losses
        avg_losses = {}
        for key in epoch_losses[0]:
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])

        return avg_losses

    def evaluate(self) -> Dict[str, float]:
        """Evaluate agent on validation environment."""
        if self.val_env is None:
            return {}
        # Show windowing information for validation environment
        try:
            if hasattr(self.val_env, "features") and self.val_env.features is not None:
                v_shape = self.val_env.features.shape
                win_size = v_shape[1]
                logger.info(
                    "Windowing: Preprocessor turns the full time series into sliding windows of size %d",
                    win_size,
                )
                logger.info("Each window is one observation for the model. Windows shape: %s", v_shape)
                sample_v = self.val_env.features[0]
                logger.info(
                    "Sample validation window[0] shape %s; first row: %s",
                    sample_v.shape,
                    np.array2string(sample_v[0], precision=4, separator=","),
                )
        except Exception:
            logger.debug("Could not log validation windows sample.")

        self.agent.eval()
        self.reward_calc.reset()

        obs = self.val_env.reset()
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(
                obs["features"]
            ).unsqueeze(0).to(self.device)

            # Match collect_rollout: fresh hidden state per window
            action, _ = self.agent.get_action(
                obs_tensor, deterministic=True
            )

            action_np = action.cpu().numpy()[0]
            obs, _, done, _ = self.val_env.step(action_np)

            if obs is None:
                break

        return self.val_env.get_performance_summary()

    def train(self) -> Dict:
        """
        Main training loop.
        
        Alternates between:
            1. Collecting rollouts with current policy
            2. Computing PPO loss and updating all parameters jointly
            3. Evaluating on validation set
            4. Logging and checkpointing
        """
        logger.info("=" * 60)
        logger.info("Starting PPO Training — Joint End-to-End Optimization")
        logger.info(f"Total timesteps: {self.total_timesteps}")
        logger.info(f"Parameters: {self.agent.count_parameters()}")
        logger.info("=" * 60)

        # Show windowing information for training environment
        try:
            if hasattr(self.train_env, "features") and self.train_env.features is not None:
                t_shape = self.train_env.features.shape
                win_size = t_shape[1]
                logger.info(
                    "Windowing: Preprocessor turns the full time series into sliding windows of size %d",
                    win_size,
                )
                logger.info("Each window is one observation for the model. Windows shape: %s", t_shape)
                sample_t = self.train_env.features[0]
                logger.info(
                    "Sample training window[0] shape %s; first row: %s",
                    sample_t.shape,
                    np.array2string(sample_t[0], precision=4, separator=","),
                )
        except Exception:
            logger.debug("Could not log training windows sample.")

        training_history = []
        start_time = time.time()

        while self.global_step < self.total_timesteps:
            # Step 1: Collect rollout
            rollout = self.collect_rollout()
            self.global_step += self.rollout_length

            # Step 2: Determine whether negative portfolio performance needs adaptive fine-tuning
            extra_epochs = 2 if self.should_adapt_on_loss(rollout) else 0
            if extra_epochs > 0:
                logger.info(
                    "Negative performance detected in recent rollout; applying %d extra fine-tuning epochs.",
                    extra_epochs,
                )

            update_metrics = self.update(rollout, extra_epochs=extra_epochs)

            # Log training metrics
            for key, value in update_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, self.global_step)

            # Rollout statistics
            avg_reward = rollout["rewards"].mean()
            avg_regime_entropy = -(
                rollout["regime_probs"] *
                np.log(rollout["regime_probs"] + 1e-10)
            ).sum(axis=-1).mean()

            self.writer.add_scalar("rollout/avg_reward", avg_reward, self.global_step)
            self.writer.add_scalar(
                "rollout/regime_entropy", avg_regime_entropy, self.global_step
            )

            # Step 3: Evaluate periodically
            if self.global_step % self.eval_frequency == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    for key, value in eval_metrics.items():
                        self.writer.add_scalar(
                            f"eval/{key}", value, self.global_step
                        )

                    # Early stopping check
                    if eval_metrics.get("sharpe_ratio", -np.inf) > self.best_sharpe:
                        self.best_sharpe = eval_metrics["sharpe_ratio"]
                        self.patience_counter = 0
                        self.save_checkpoint("best_model.pt")
                        logger.info(
                            f"New best Sharpe: {self.best_sharpe:.4f}"
                        )
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.early_stopping_patience:
                            logger.info("Early stopping triggered.")
                            break

                elapsed = time.time() - start_time
                logger.info(
                    f"Step {self.global_step}/{self.total_timesteps} | "
                    f"Loss: {update_metrics['total_loss']:.4f} | "
                    f"Reward: {avg_reward:.4f} | "
                    f"Sharpe: {eval_metrics.get('sharpe_ratio', 'N/A')} | "
                    f"Time: {elapsed:.0f}s"
                )

            # Step 4: Save checkpoint
            if self.global_step % self.save_frequency == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step}.pt")

            training_history.append({
                "step": self.global_step,
                **update_metrics,
                "avg_reward": avg_reward,
            })

        # Final save
        self.save_checkpoint("final_model.pt")
        self.writer.close()

        logger.info("Training complete!")
        return {"history": training_history}

    def should_adapt_on_loss(self, rollout: Dict) -> bool:
        """Return True when a recent rollout suffered negative portfolio returns."""
        portfolio_returns = rollout.get("portfolio_returns")
        if portfolio_returns is None or len(portfolio_returns) == 0:
            return False

        avg_return = float(np.mean(portfolio_returns))
        worst_return = float(np.min(portfolio_returns))
        return avg_return < 0.0 or worst_return < -0.02

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_sharpe": self.best_sharpe,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        # checkpoint = torch.load(path, map_location=self.device)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_sharpe = checkpoint["best_sharpe"]
        logger.info(f"Checkpoint loaded: {path}, step {self.global_step}")
