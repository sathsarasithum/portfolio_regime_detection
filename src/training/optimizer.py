"""
Joint End-to-End Optimizer Setup
=================================
Configures the joint optimizer for end-to-end PPO training,
where gradients flow through Encoder → Actor → Critic simultaneously.
"""

import torch
import torch.optim as optim
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    encoder_lr_scale: float = 0.5,
) -> torch.optim.Optimizer:
    """
    Create optimizer with per-component learning rate scheduling.
    
    The regime encoder uses a lower learning rate since it's deeper
    and needs more stable updates for the joint optimization.
    
    Args:
        model: PPO Agent model
        learning_rate: base learning rate
        weight_decay: L2 regularization weight
        encoder_lr_scale: learning rate multiplier for the encoder
        
    Returns:
        optimizer: AdamW optimizer with parameter groups
    """
    # Separate parameters by component for differential learning rates
    param_groups = [
        {
            "params": model.regime_encoder.parameters(),
            "lr": learning_rate * encoder_lr_scale,
            "name": "regime_encoder",
        },
        {
            "params": model.actor.parameters(),
            "lr": learning_rate,
            "name": "actor",
        },
        {
            "params": model.critic.parameters(),
            "lr": learning_rate,
            "name": "critic",
        },
    ]

    optimizer = optim.AdamW(
        param_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: optimizer to schedule
        total_steps: total number of training steps
        warmup_steps: number of linear warmup steps
        min_lr_ratio: minimum LR as fraction of initial LR
        
    Returns:
        scheduler: LR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            import math
            progress = (current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(
                min_lr_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
