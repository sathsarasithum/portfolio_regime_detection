"""
Preprocessing Module
====================
Windowing, feature engineering, and normalization.
Corresponds to the "Preprocessing Module (Windowing, Feature Engineering)" block.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Transforms raw price/volume data into feature-engineered temporal contexts.
    
    Pipeline:
        Raw Prices/Volumes → Returns → Technical Indicators → Normalization → Windowing → x_t
    """

    # Fixed per-asset suffix order produced by engineer_features(). Every asset
    # contributes exactly one column per suffix below (with "" meaning the raw
    # log-return column itself, named after the asset with no suffix).
    PER_ASSET_SUFFIXES = [
        "", "_vol_5d", "_vol_10d", "_vol_20d",
        "_mom_5d", "_mom_10d", "_mom_20d",
        "_rsi", "_macd", "_macd_signal", "_macd_hist", "_bb_pct",
        "_vol_ratio_5d", "_vol_ratio_10d", "_vol_ratio_20d",
    ]

    def __init__(self, window_size: int = 60, normalize_method: str = "zscore"):
        self.window_size = window_size
        self.normalize_method = normalize_method
        self.scalers: Dict[str, object] = {}

    # ──────────────────────────────────────────────
    # Feature Engineering
    # ──────────────────────────────────────────────
    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns: ln(P_t / P_{t-1})."""
        returns = np.log(prices / prices.shift(1))
        return returns.iloc[1:]  # Drop first NaN row

    def compute_volatility(self, returns: pd.DataFrame,
                           windows: list = [5, 10, 20]) -> pd.DataFrame:
        """Rolling realized volatility at multiple horizons."""
        vol_features = pd.DataFrame(index=returns.index)
        for w in windows:
            vol = returns.rolling(window=w).std() * np.sqrt(252)
            vol.columns = [f"{col}_vol_{w}d" for col in returns.columns]
            vol_features = pd.concat([vol_features, vol], axis=1)
        return vol_features

    def compute_momentum(self, prices: pd.DataFrame,
                         windows: list = [5, 10, 20]) -> pd.DataFrame:
        """Price momentum (rate of change) at multiple horizons."""
        mom_features = pd.DataFrame(index=prices.index)
        for w in windows:
            mom = prices.pct_change(periods=w)
            mom.columns = [f"{col}_mom_{w}d" for col in prices.columns]
            mom_features = pd.concat([mom_features, mom], axis=1)
        return mom_features

    def compute_rsi(self, prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Relative Strength Index."""
        rsi_features = pd.DataFrame(index=prices.index)
        for col in prices.columns:
            delta = prices[col].diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            rsi_features[f"{col}_rsi"] = 100 - (100 / (1 + rs))
        return rsi_features

    def compute_macd(self, prices: pd.DataFrame,
                     fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD indicator."""
        col_data = {}
        for col in prices.columns:
            ema_fast = prices[col].ewm(span=fast, adjust=False).mean()
            ema_slow = prices[col].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            col_data[f"{col}_macd"] = macd_line
            col_data[f"{col}_macd_signal"] = signal_line
            col_data[f"{col}_macd_hist"] = macd_line - signal_line
        return pd.DataFrame(col_data, index=prices.index)

    def compute_volume_features(self, volumes: pd.DataFrame,
                                windows: list = [5, 10, 20]) -> pd.DataFrame:
        """Volume-based features: rolling mean ratios, OBV-inspired."""
        vol_features = pd.DataFrame(index=volumes.index)
        for w in windows:
            vol_ma = volumes.rolling(window=w).mean()
            vol_ratio = volumes / (vol_ma + 1e-10)
            vol_ratio.columns = [f"{col}_vol_ratio_{w}d" for col in volumes.columns]
            vol_features = pd.concat([vol_features, vol_ratio], axis=1)
        return vol_features

    def compute_bollinger_bands(self, prices: pd.DataFrame,
                                window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Bollinger Band %B indicator."""
        bb_features = pd.DataFrame(index=prices.index)
        for col in prices.columns:
            sma = prices[col].rolling(window=window).mean()
            std = prices[col].rolling(window=window).std()
            upper = sma + num_std * std
            lower = sma - num_std * std
            bb_features[f"{col}_bb_pct"] = (prices[col] - lower) / (upper - lower + 1e-10)
        return bb_features

    # ──────────────────────────────────────────────
    # Full Feature Engineering Pipeline
    # ──────────────────────────────────────────────
    def engineer_features(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None,
        use_returns: bool = True,
        use_volatility: bool = True,
        use_momentum: bool = True,
        use_volume_features: bool = True,
    ) -> pd.DataFrame:
        """
        Full feature engineering pipeline.
        
        Returns:
            DataFrame with all engineered features, aligned by date.
        """
        feature_frames = []

        # 1. Log returns
        if use_returns:
            returns = self.compute_log_returns(prices)
            feature_frames.append(returns)

        # 2. Rolling volatility
        if use_volatility:
            returns_for_vol = self.compute_log_returns(prices) if not use_returns else returns
            vol = self.compute_volatility(returns_for_vol)
            feature_frames.append(vol)

        # 3. Momentum
        if use_momentum:
            mom = self.compute_momentum(prices)
            feature_frames.append(mom)

        # 4. RSI
        rsi = self.compute_rsi(prices)
        feature_frames.append(rsi)

        # 5. MACD
        macd = self.compute_macd(prices)
        feature_frames.append(macd)

        # 6. Bollinger Bands
        bb = self.compute_bollinger_bands(prices)
        feature_frames.append(bb)

        # 7. Volume features
        if use_volume_features and volumes is not None:
            vol_feats = self.compute_volume_features(volumes)
            feature_frames.append(vol_feats)

        # Concatenate all features
        all_features = pd.concat(feature_frames, axis=1)

        # 8. Macroeconomic data (if available)
        if macro_data is not None:
            if "Date" in macro_data.columns:
                macro_data = macro_data.set_index("Date")
            macro_data.index = pd.to_datetime(macro_data.index)
            # Forward fill macro data to daily frequency
            macro_data = macro_data.reindex(all_features.index, method="ffill")
            all_features = pd.concat([all_features, macro_data], axis=1)

        # Drop NaN rows from rolling computations
        all_features = all_features.dropna()

        logger.info(
            f"Engineered features: {all_features.shape[1]} features, "
            f"{all_features.shape[0]} timesteps"
        )
        return all_features

    # ──────────────────────────────────────────────
    # Per-asset column reordering (for the per-asset encoder)
    # ──────────────────────────────────────────────
    def reorder_per_asset(
        self,
        features: pd.DataFrame,
        asset_names: list,
    ) -> pd.DataFrame:
        """
        Reorder engineer_features() output so each asset's PER_ASSET_SUFFIXES
        columns are contiguous, in identical order across assets:
            [asset0_f0, asset0_f1, ..., asset0_f14, asset1_f0, ..., assetN_f14]

        This lets the model reshape the flat (T, N*F) feature matrix into
        (T, N, F) and process each asset with a shared per-asset encoder,
        instead of mixing all assets' raw features into one vector.

        Args:
            features: DataFrame from engineer_features() (any column order).
                Must NOT include macro columns — those aren't per-asset.
            asset_names: list of N asset tickers, in the desired node order.

        Returns:
            DataFrame with the same rows, reordered/selected columns:
            shape (T, N * len(PER_ASSET_SUFFIXES)).
        """
        cols = []
        for asset in asset_names:
            for suffix in self.PER_ASSET_SUFFIXES:
                col = f"{asset}{suffix}"
                if col not in features.columns:
                    raise KeyError(
                        f"Expected per-asset column '{col}' not found in features. "
                        f"reorder_per_asset() assumes engineer_features() was called "
                        f"with default settings (all indicator blocks enabled) and "
                        f"no macro_data (macro columns aren't per-asset)."
                    )
                cols.append(col)

        logger.info("Reordering per-asset columns for assets: %s", asset_names)
        logger.info("Per-asset column order (%d columns): %s", len(cols), cols)
        print(f"[reorder_per_asset] assets={asset_names}")
        print(f"[reorder_per_asset] columns={cols}")

        return features[cols]

    @property
    def n_features_per_asset(self) -> int:
        """Number of engineered features per asset (columns in PER_ASSET_SUFFIXES)."""
        return len(self.PER_ASSET_SUFFIXES)

    # ──────────────────────────────────────────────
    # Normalization
    # ──────────────────────────────────────────────
    def fit_normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on training data and transform."""
        if self.normalize_method == "zscore":
            scaler = StandardScaler()
        elif self.normalize_method == "minmax":
            scaler = MinMaxScaler()
        elif self.normalize_method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization: {self.normalize_method}")

        normalized = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        self.scalers["features"] = scaler
        return normalized

    def transform_normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform using already-fitted scaler."""
        scaler = self.scalers.get("features")
        if scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_normalize first.")

        return pd.DataFrame(
            scaler.transform(features),
            index=features.index,
            columns=features.columns
        )

    # ──────────────────────────────────────────────
    # Windowing → x_t
    # ──────────────────────────────────────────────
    def create_windows(self, features: np.ndarray) -> np.ndarray:
        """
        Create sliding windows of temporal context.
        
        Args:
            features: (T, F) array of normalized features
            
        Returns:
            windows: (T - window_size + 1, window_size, F) array
        """
        T, F = features.shape
        n_windows = T - self.window_size + 1
        windows = np.zeros((n_windows, self.window_size, F))
        for i in range(n_windows):
            windows[i] = features[i : i + self.window_size]
        return windows

    # ──────────────────────────────────────────────
    # Full Pipeline
    # ──────────────────────────────────────────────
    def fit_transform(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame, pd.DatetimeIndex]:
        """
        Complete preprocessing pipeline: features → normalize → window.
        
        Returns:
            windows: (N, window_size, F) temporal context arrays
            raw_features: DataFrame of all features before windowing
            dates: DatetimeIndex aligned with windows
        """
        # Engineer features
        features = self.engineer_features(prices, volumes, macro_data)

        # Normalize
        normalized = self.fit_normalize(features)

        # Window
        windows = self.create_windows(normalized.values)

        # Align dates (each window's label is its last date)
        dates = features.index[self.window_size - 1:]

        logger.info(
            f"Preprocessing complete: {windows.shape[0]} windows of shape "
            f"({self.window_size}, {windows.shape[2]})"
        )

        return windows, features, dates
