"""
CSE Data Loader
===============
Loads raw Colombo Stock Exchange price and volume data.
Handles dynamic header detection, sheet concatenation, cleaning, and pivoting.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class CSEDataLoader:
    """
    Loads and merges raw CSE market data files from raw sheets.
    Handles dynamic header detection, sheet concatenation, and pivoting.
    """

    def __init__(self, raw_data_dir: str, macro_data_dir: Optional[str] = None):
        self.raw_data_dir = raw_data_dir
        self.macro_data_dir = macro_data_dir

    def load_raw_cse_data(self) -> pd.DataFrame:
        """
        Scan and load all Excel/CSV files in the raw data directory.
        Uses dynamic header detection to parse sheets.
        
        Returns:
            Concatenated DataFrame of all loaded raw data.
        """
        all_dfs = []
        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")

        files = sorted(os.listdir(self.raw_data_dir))
        for filename in files:
            filepath = os.path.join(self.raw_data_dir, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext in [".xls", ".xlsx"]:
                logger.info(f"Loading Excel file: {filename}")
                try:
                    xl = pd.ExcelFile(filepath)
                    for sheet in xl.sheet_names:
                        # Load first 10 rows to detect header row
                        df_sample = pd.read_excel(filepath, sheet_name=sheet, header=None, nrows=10)
                        header_row = None
                        for i in range(len(df_sample)):
                            row_vals = [str(x).strip().upper() for x in df_sample.iloc[i].values]
                            if any("COMPANY ID" in x or "COMPANY CODE" in x or "COMPANY" in x for x in row_vals):
                                header_row = i
                                break

                        if header_row is None:
                            logger.warning(f"Header not found in sheet '{sheet}' of {filename}. Skipping.")
                            continue

                        # Load full sheet starting from header_row
                        df_sheet = pd.read_excel(filepath, sheet_name=sheet, skiprows=header_row)
                        df_sheet.columns = [str(c).strip() for c in df_sheet.columns]
                        all_dfs.append(df_sheet)
                        logger.info(f"Loaded sheet '{sheet}': shape={df_sheet.shape}")
                except Exception as e:
                    logger.error(f"Error loading Excel file {filename}: {e}")

            elif ext == ".csv":
                logger.info(f"Loading CSV file: {filename}")
                try:
                    df_sample = pd.read_csv(filepath, header=None, nrows=10)
                    header_row = None
                    for i in range(len(df_sample)):
                        row_vals = [str(x).strip().upper() for x in df_sample.iloc[i].values]
                        if any("COMPANY ID" in x or "COMPANY CODE" in x or "COMPANY" in x for x in row_vals):
                            header_row = i
                            break

                    if header_row is None:
                        logger.warning(f"Header not found in CSV {filename}. Skipping.")
                        continue

                    df_csv = pd.read_csv(filepath, skiprows=header_row)
                    df_csv.columns = [str(c).strip() for c in df_csv.columns]
                    all_dfs.append(df_csv)
                    logger.info(f"Loaded CSV: shape={df_csv.shape}")
                except Exception as e:
                    logger.error(f"Error loading CSV file {filename}: {e}")

        if not all_dfs:
            raise ValueError(f"No valid data sheets could be loaded from {self.raw_data_dir}")

        # Concatenate all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined raw data shape: {combined_df.shape}")
        return combined_df

    def load_prices_and_volumes(
        self,
        n_assets: int = 0,
        max_missing_pct: float = 0.05,
        cache_dir: Optional[str] = None,
        force_reload: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes raw CSE data and returns aligned price and volume matrices.

        On first call, parses raw Excel/CSV files (slow ~90s) and saves
        parquet cache files to cache_dir. Subsequent calls load from cache (~1s).

        Args:
            n_assets: Number of top assets to keep based on completeness.
                      If 0, keeps all assets with missing percentage <= max_missing_pct.
            max_missing_pct: Maximum percentage of missing values allowed if n_assets is 0.
            cache_dir: Directory to store/read parquet caches. Defaults to raw_data_dir/../processed.
            force_reload: If True, ignore the cache and re-parse raw files.

        Returns:
            price_matrix: DataFrame of shape (Dates x Assets) of closing prices.
            volume_matrix: DataFrame of shape (Dates x Assets) of volume.
        """
        # ── Cache handling ──────────────────────────────────────────────
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(self.raw_data_dir), "processed")
        os.makedirs(cache_dir, exist_ok=True)

        cache_suffix = f"n{n_assets}" if n_assets > 0 else f"mp{int(max_missing_pct*100)}"
        price_cache = os.path.join(cache_dir, f"price_matrix_{cache_suffix}.parquet")
        volume_cache = os.path.join(cache_dir, f"volume_matrix_{cache_suffix}.parquet")

        if not force_reload and os.path.exists(price_cache) and os.path.exists(volume_cache):
            logger.info(f"Loading price/volume matrices from cache: {cache_dir}")
            price_matrix = pd.read_parquet(price_cache)
            volume_matrix = pd.read_parquet(volume_cache)
            logger.info(f"Cached price matrix shape: {price_matrix.shape}")
            return price_matrix, volume_matrix

        combined_df = self.load_raw_cse_data()

        # Rename columns to standard ones if there are subtle differences
        col_mapping = {}
        for col in combined_df.columns:
            col_upper = str(col).strip().upper()
            if col_upper in ["COMPANY ID", "COMPANY ID (RS.)"]:
                col_mapping[col] = "COMPANY ID"
            elif col_upper in ["TRADING DATE", "DATE"]:
                col_mapping[col] = "TRADING DATE"
            elif col_upper in ["CLOSE PRICE (RS.)", "CLOSE PRICE", "CLOSE"]:
                col_mapping[col] = "CLOSE PRICE (Rs.)"
            elif col_upper in ["SHARE VOLUME (NO.)", "SHARE VOLUME", "VOLUME", "VOLUME (NO.)"]:
                col_mapping[col] = "SHARE VOLUME (No.)"
            # Fallbacks if exact matches aren't found
            elif "COMPANY ID" in col_upper or "COMPANY CODE" in col_upper:
                col_mapping[col] = "COMPANY ID"
            elif "TRADING DATE" in col_upper:
                col_mapping[col] = "TRADING DATE"
            elif "CLOSE PRICE" in col_upper:
                col_mapping[col] = "CLOSE PRICE (Rs.)"
            elif "SHARE VOLUME" in col_upper:
                col_mapping[col] = "SHARE VOLUME (No.)"

        combined_df = combined_df.rename(columns=col_mapping)
        
        # Keep only the first occurrence of any duplicate columns to prevent DataFrame output on selection
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        required_cols = ["COMPANY ID", "TRADING DATE", "CLOSE PRICE (Rs.)", "SHARE VOLUME (No.)"]
        for col in required_cols:
            if col not in combined_df.columns:
                raise ValueError(f"Required column '{col}' is missing from loaded data. Available: {combined_df.columns.tolist()}")

        # Drop rows where COMPANY ID or TRADING DATE is null
        combined_df = combined_df.dropna(subset=["COMPANY ID", "TRADING DATE"])
        combined_df["COMPANY ID"] = combined_df["COMPANY ID"].astype(str).str.strip()
        combined_df["TRADING DATE"] = pd.to_datetime(combined_df["TRADING DATE"], format="mixed", errors="coerce")
        combined_df = combined_df.dropna(subset=["TRADING DATE"])

        # Clean numeric values
        combined_df["CLOSE PRICE (Rs.)"] = pd.to_numeric(
            combined_df["CLOSE PRICE (Rs.)"].astype(str).str.replace(",", ""), errors="coerce"
        )
        combined_df["SHARE VOLUME (No.)"] = pd.to_numeric(
            combined_df["SHARE VOLUME (No.)"].astype(str).str.replace(",", ""), errors="coerce"
        )

        # Drop rows with NaN in numeric columns
        combined_df = combined_df.dropna(subset=["CLOSE PRICE (Rs.)", "SHARE VOLUME (No.)"])

        # Pivot to Dates x Companies
        logger.info("Pivoting data to matrices...")
        price_pivot = combined_df.pivot_table(
            index="TRADING DATE", columns="COMPANY ID", values="CLOSE PRICE (Rs.)", aggfunc="last"
        )
        volume_pivot = combined_df.pivot_table(
            index="TRADING DATE", columns="COMPANY ID", values="SHARE VOLUME (No.)", aggfunc="last"
        )

        # Align index
        common_dates = price_pivot.index.intersection(volume_pivot.index)
        price_pivot = price_pivot.loc[common_dates].sort_index()
        volume_pivot = volume_pivot.loc[common_dates].sort_index()

        # Filter assets based on missingness
        missing_pct = price_pivot.isna().mean()
        
        if n_assets > 0:
            # Keep top n_assets with the least missing values
            selected_assets = missing_pct.sort_values().index[:n_assets].tolist()
            logger.info(f"Selecting top {n_assets} assets based on lowest missingness.")
        else:
            # Keep assets with missingness <= max_missing_pct
            selected_assets = missing_pct[missing_pct <= max_missing_pct].index.tolist()
            logger.info(f"Selecting {len(selected_assets)} assets with missingness <= {max_missing_pct * 100}%.")

        if not selected_assets:
            raise ValueError("No assets met the missingness threshold criteria.")

        price_matrix = price_pivot[selected_assets].copy()
        volume_matrix = volume_pivot[selected_assets].copy()

        # Handle missing values: forward fill and backward fill
        price_matrix = price_matrix.ffill().bfill()
        # For volume, forward fill first, then fill remaining NaNs with 0
        volume_matrix = volume_matrix.ffill().fillna(0.0)

        # Final sanity check: ensure no NaNs are left
        if price_matrix.isna().sum().sum() > 0 or volume_matrix.isna().sum().sum() > 0:
            logger.warning("NaNs are still present in matrices after filling. Dropping remaining NaN rows.")
            price_matrix = price_matrix.dropna()
            volume_matrix = volume_matrix.loc[price_matrix.index]

        logger.info(f"Final aligned price matrix shape: {price_matrix.shape}")
        logger.info(f"Final aligned volume matrix shape: {volume_matrix.shape}")

        # ── Save to parquet cache for fast reloads ──────────────────────
        try:
            price_matrix.to_parquet(price_cache)
            volume_matrix.to_parquet(volume_cache)
            logger.info(f"Saved matrices to cache: {cache_dir}")
        except Exception as e:
            logger.warning(f"Could not save cache (continuing): {e}")

        return price_matrix, volume_matrix
