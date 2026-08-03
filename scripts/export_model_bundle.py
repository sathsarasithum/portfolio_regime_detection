"""
One-time export step — run AFTER training finishes.

The training loop (src/training/trainer.py) saves the model checkpoint but
never persists the fitted feature scaler, so there's currently no way to
normalize new (live) data the same way the model saw it during training.

This script rebuilds the exact train-split scaler main.py fits (same
loader, same feature engineering, same train_ratio — fit on the TRAIN
split only, never refit later) and writes it, together with a manifest
describing the trained architecture and asset universe, into
results/models/. backend/inference.py requires this bundle to serve live
predictions.

Usage (from the project root, after training):
    python scripts/export_model_bundle.py                        # auto-picks a checkpoint
    python scripts/export_model_bundle.py --checkpoint final_model.pt
"""
import argparse
import json
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import DATA_RAW_DIR, MODELS_DIR, get_config
from src.data.data_loader import CSEDataLoader
from src.data.preprocessor import Preprocessor

# Preference order when --checkpoint isn't given: best_model.pt (validation
# -selected) if it exists, else final_model.pt (whatever training ended on).
CHECKPOINT_PREFERENCE = ["best_model.pt", "final_model.pt"]


def resolve_checkpoint(explicit):
    if explicit:
        if not os.path.exists(os.path.join(MODELS_DIR, explicit)):
            raise FileNotFoundError(f"{os.path.join(MODELS_DIR, explicit)} not found.")
        return explicit

    for name in CHECKPOINT_PREFERENCE:
        if os.path.exists(os.path.join(MODELS_DIR, name)):
            return name

    raise FileNotFoundError(
        f"No checkpoint found in {MODELS_DIR} (looked for: {CHECKPOINT_PREFERENCE}). "
        f"Train the model first (python main.py --mode train)."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint filename in results/models/ (default: auto-pick best_model.pt "
             "if present, else final_model.pt).",
    )
    args = parser.parse_args()

    config = get_config()
    config.data.n_macro_variables = 0

    checkpoint_filename = resolve_checkpoint(args.checkpoint)
    checkpoint_path = os.path.join(MODELS_DIR, checkpoint_filename)
    print(f"Using checkpoint: {checkpoint_path}")

    loader = CSEDataLoader(raw_data_dir=DATA_RAW_DIR)
    prices, volumes = loader.load_prices_and_volumes(n_assets=47)
    asset_names = list(prices.columns)
    config.data.asset_names = asset_names
    config.data.n_assets = len(asset_names)

    preprocessor = Preprocessor(
        window_size=config.data.window_size,
        normalize_method=config.data.normalize_method,
    )
    features_all = preprocessor.engineer_features(prices, volumes, macro_data=None)
    features_all = preprocessor.reorder_per_asset(features_all, asset_names)

    n_total = len(features_all)
    n_train = int(n_total * config.data.train_ratio)
    features_train = features_all.iloc[:n_train]

    # Fit ONLY on the train split — must match main.py exactly, or the
    # model will see inputs normalized differently than it was trained on.
    preprocessor.fit_normalize(features_train)
    scaler = preprocessor.scalers["features"]

    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    manifest = {
        "checkpoint_filename": checkpoint_filename,
        "asset_names": asset_names,
        "n_assets": config.data.n_assets,
        "n_features_per_asset": preprocessor.n_features_per_asset,
        "window_size": config.data.window_size,
        "vsn_hidden_dim": config.regime_encoder.vsn_hidden_dim,
        "lstm_hidden_dim": config.regime_encoder.lstm_hidden_dim,
        "lstm_num_layers": config.regime_encoder.lstm_num_layers,
        "mha_num_heads": config.regime_encoder.mha_num_heads,
        "gnn_hidden_dim": config.regime_encoder.gnn_hidden_dim,
        "gnn_num_heads": config.regime_encoder.gnn_num_heads,
        "gnn_num_layers": config.regime_encoder.gnn_num_layers,
        "n_regimes": config.regime_encoder.n_regimes,
        "latent_state_dim": config.regime_encoder.latent_state_dim,
        "actor_hidden_dims": config.actor.hidden_dims,
        "critic_hidden_dims": config.critic.hidden_dims,
        "allow_short": config.environment.allow_short_selling,
        "max_position_size": config.environment.max_position_size,
        "min_trading_days_required": 60,
    }
    manifest_path = os.path.join(MODELS_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved scaler   -> {scaler_path}")
    print(f"Saved manifest -> {manifest_path}")
    print("Model bundle ready for backend/inference.py (GET /api/live/today).")


if __name__ == "__main__":
    main()
