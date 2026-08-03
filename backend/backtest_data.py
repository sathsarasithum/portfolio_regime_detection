"""
Reconstructs the real chronological test-set date index (the exact split
main.py produces) and joins it with the already-exported
results/test_weights.csv (per-day weights from the promoted model's
test-set backtest) and results/test_summary.json (summary metrics).

No model/torch is needed for this — it's the same deterministic pandas
feature-engineering + split logic as main.py, just re-run to recover the
calendar dates that weren't saved alongside the weights export.
"""
import functools
import json
import os

import pandas as pd

from .config import (
    ASSET_DISPLAY_NAMES,
    DATA_RAW_DIR,
    TEST_SUMMARY_PATH,
    TEST_WEIGHTS_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    WINDOW_SIZE,
)
from src.data.data_loader import CSEDataLoader
from src.data.preprocessor import Preprocessor


@functools.lru_cache(maxsize=1)
def get_backtest_dataset() -> dict:
    if not os.path.exists(TEST_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"{TEST_WEIGHTS_PATH} not found — run `python main.py --mode train` "
            f"(or --mode backtest) first to produce it."
        )
    if not os.path.exists(TEST_SUMMARY_PATH):
        raise FileNotFoundError(
            f"{TEST_SUMMARY_PATH} not found — run `python main.py --mode train` "
            f"(or --mode backtest) first to produce it."
        )

    loader = CSEDataLoader(raw_data_dir=DATA_RAW_DIR)
    prices, volumes = loader.load_prices_and_volumes(n_assets=47)
    asset_names = list(prices.columns)

    preprocessor = Preprocessor(window_size=WINDOW_SIZE, normalize_method="zscore")
    features_all = preprocessor.engineer_features(prices, volumes, macro_data=None)
    features_all = preprocessor.reorder_per_asset(features_all, asset_names)
    prices = prices.loc[features_all.index]

    n_total = len(features_all)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    features_test = features_all.iloc[n_train + n_val:]
    prices_test = prices.loc[features_test.index]
    test_dates = prices_test.index[WINDOW_SIZE - 1:]

    weights_df = pd.read_csv(TEST_WEIGHTS_PATH)

    # The exported weight rows and the reconstructed date index should line
    # up 1:1 (row i's action was decided from the window ending test_dates[i-1]
    # and applied over test_dates[i-1] -> test_dates[i]); tolerate a possible
    # off-by-one from the environment's stopping condition rather than crash.
    n_days = min(len(test_dates), len(weights_df))
    weights_df = weights_df.iloc[:n_days].reset_index(drop=True)
    dates = test_dates[:n_days]

    records = []
    for i in range(n_days):
        row = weights_df.iloc[i]
        portfolio_value = float(row["portfolio_value"])
        allocations = []
        for asset in asset_names:
            col = f"{asset}_weight"
            if col not in weights_df.columns:
                continue
            weight = float(row[col])
            allocations.append(
                {
                    "asset": asset,
                    "name": ASSET_DISPLAY_NAMES.get(asset, asset),
                    "weight": weight,
                    "value_lkr": weight * portfolio_value,
                }
            )
        allocations.sort(key=lambda a: a["weight"], reverse=True)

        records.append(
            {
                "index": i,
                "date": dates[i].strftime("%Y-%m-%d"),
                "decision_basis_date": dates[i - 1].strftime("%Y-%m-%d") if i > 0 else None,
                "portfolio_value": portfolio_value,
                "portfolio_return": float(row["portfolio_return"]),
                "total_cost": float(row["total_cost"]),
                "allocations": allocations,
            }
        )

    with open(TEST_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return {
        "asset_names": asset_names,
        "days": records,
        "summary": summary,
    }
