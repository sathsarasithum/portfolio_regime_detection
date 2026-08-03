"""
Shared inference pipeline: trailing daily price/volume history -> model
weights + regime probabilities for a single "as of today" decision.

Mirrors main.py's preprocessing exactly (Preprocessor.engineer_features ->
reorder_per_asset -> scaler.transform -> create_windows -> PPOAgent ->
clip weights), but for one trailing window instead of a full chronological
split, and using an already-fitted (never refit) scaler.

Requires a model bundle — checkpoint + scaler + manifest — produced by
scripts/export_model_bundle.py, which must be run once after training
finishes. Nothing here is usable until that bundle exists.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

from .config import MANIFEST_PATH, MODELS_DIR, SCALER_PATH
from src.data.preprocessor import Preprocessor
from src.models.ppo_agent import PPOAgent


class ModelNotReadyError(RuntimeError):
    """Raised when the checkpoint/scaler/manifest bundle isn't available yet."""


_bundle_cache: dict = {}


def load_bundle():
    """Loads (and caches for the process lifetime) the agent, scaler, and
    manifest. Raises ModelNotReadyError with a clear remedy if any part of
    the bundle is missing."""
    if "bundle" in _bundle_cache:
        return _bundle_cache["bundle"]

    missing = [p for p in (SCALER_PATH, MANIFEST_PATH) if not os.path.exists(p)]
    if missing:
        raise ModelNotReadyError(
            "Model bundle incomplete — missing: " + ", ".join(missing) + ". "
            "Run `python scripts/export_model_bundle.py` once training finishes."
        )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    checkpoint_path = os.path.join(MODELS_DIR, manifest["checkpoint_filename"])
    if not os.path.exists(checkpoint_path):
        raise ModelNotReadyError(
            f"Manifest points at checkpoint '{manifest['checkpoint_filename']}' but "
            f"{checkpoint_path} doesn't exist. Re-run scripts/export_model_bundle.py."
        )

    agent = PPOAgent(
        n_features_per_asset=manifest["n_features_per_asset"],
        n_assets=manifest["n_assets"],
        n_macro=0,
        vsn_hidden_dim=manifest["vsn_hidden_dim"],
        lstm_hidden_dim=manifest["lstm_hidden_dim"],
        lstm_num_layers=manifest["lstm_num_layers"],
        mha_num_heads=manifest["mha_num_heads"],
        gnn_hidden_dim=manifest["gnn_hidden_dim"],
        gnn_num_heads=manifest["gnn_num_heads"],
        gnn_num_layers=manifest["gnn_num_layers"],
        n_regimes=manifest["n_regimes"],
        latent_state_dim=manifest["latent_state_dim"],
        actor_hidden_dims=manifest["actor_hidden_dims"],
        critic_hidden_dims=manifest["critic_hidden_dims"],
        allow_short=manifest["allow_short"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    scaler = joblib.load(SCALER_PATH)

    _bundle_cache["bundle"] = (agent, scaler, manifest)
    return _bundle_cache["bundle"]


def _clip_weights(weights: np.ndarray, max_position_size: float, allow_short: bool) -> np.ndarray:
    """Mirrors MarketEnvironment._clip_weights (src/environment/market_env.py)
    so served weights match validated backtest behavior exactly."""
    if max_position_size < 1.0:
        weights = np.clip(weights, -max_position_size, max_position_size)
        if not allow_short:
            total = weights.sum()
            if total > 0:
                weights = weights / total
    return weights


def predict_weights(prices: pd.DataFrame, volumes: pd.DataFrame) -> dict:
    """
    prices/volumes: (T, n_assets) trailing daily history, columns matching
    manifest['asset_names'], T >= manifest['min_trading_days_required'].
    """
    agent, scaler, manifest = load_bundle()
    asset_names = manifest["asset_names"]

    preprocessor = Preprocessor(window_size=manifest["window_size"], normalize_method="zscore")
    features = preprocessor.engineer_features(prices, volumes, macro_data=None)
    features = preprocessor.reorder_per_asset(features, asset_names)

    if len(features) < manifest["window_size"]:
        raise ValueError(
            f"Only {len(features)} usable rows after feature engineering "
            f"(rolling/EMA warm-up consumes the first ~26 rows) — need at "
            f"least {manifest['window_size']}."
        )

    normalized = pd.DataFrame(
        scaler.transform(features), index=features.index, columns=features.columns
    )
    window = normalized.values[-manifest["window_size"]:]
    obs = torch.FloatTensor(window).unsqueeze(0)

    with torch.no_grad():
        action, info = agent.get_action(obs, deterministic=True)

    weights = _clip_weights(
        action.cpu().numpy()[0], manifest["max_position_size"], manifest["allow_short"]
    )
    regime = info["regime_probs"].cpu().numpy()[0]

    return {
        "as_of_date": str(features.index[-1].date()),
        "weights": {asset: float(w) for asset, w in zip(asset_names, weights)},
        "regime_probs": {
            label: float(p) for label, p in zip(["bull", "bear", "sideways"], regime)
        },
    }
