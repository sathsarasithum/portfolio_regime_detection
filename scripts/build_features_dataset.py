"""
Build the full engineered-feature dataset from the merged CSE banking
sector CSV and save it to data/processed/.

Pipeline: merged raw CSV -> price/volume matrices -> Preprocessor
feature engineering (15 features per asset, unnormalized) -> CSV.

The 15 features computed per asset are:
    log_return, vol_5d, vol_10d, vol_20d, mom_5d, mom_10d, mom_20d,
    rsi, macd, macd_signal, macd_hist, bb_pct,
    vol_ratio_5d, vol_ratio_10d, vol_ratio_20d

Features are saved unnormalized (raw values) since normalization should
be fit on the training split only, at train time, to avoid look-ahead
leakage into validation/test data.

Usage:
    python scripts/build_features_dataset.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, get_config
from src.data.data_loader import CSEDataLoader
from src.data.preprocessor import Preprocessor

OUTPUT_PATH = os.path.join(DATA_PROCESSED_DIR, "banking_sector_features.csv")


def main() -> None:
    config = get_config()

    loader = CSEDataLoader(raw_data_dir=DATA_RAW_DIR)
    price_matrix, volume_matrix = loader.load_prices_and_volumes(
        max_missing_pct=0.05, force_reload=True
    )
    print(f"Price matrix: {price_matrix.shape}, assets: {price_matrix.columns.tolist()}")

    preprocessor = Preprocessor(
        window_size=config.data.window_size,
        normalize_method=config.data.normalize_method,
    )
    features = preprocessor.engineer_features(
        price_matrix,
        volume_matrix,
        use_returns=config.data.use_returns,
        use_volatility=config.data.use_volatility,
        use_momentum=config.data.use_momentum,
        use_volume_features=config.data.use_volume_features,
    )

    n_assets = len(price_matrix.columns)
    features_per_asset = features.shape[1] / n_assets
    print(f"Engineered features: {features.shape[1]} columns "
          f"({features_per_asset:.0f} per asset x {n_assets} assets), "
          f"{features.shape[0]} rows")

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index_label="TRADING DATE")
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
