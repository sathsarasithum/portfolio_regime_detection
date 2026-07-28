"""
Merge yearly CSE banking sector CSV files (2021-2025) in data/raw/ into
a single chronologically sorted CSV.

Handles inconsistencies observed across the yearly files:
  - Date formats differ per year (e.g. "04-JAN-21", "2-Jan-23",
    "2025-01-02 00:00:00") -> parsed and normalized to YYYY-MM-DD.
  - 2023 file has trailing empty "Unnamed" columns -> dropped.
  - 2023 file has thousands-separators in numeric columns as quoted
    strings (e.g. "85,401") -> commas stripped and cast to numeric.

Usage:
    python scripts/merge_raw_data.py
"""

import glob
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_PATH = os.path.join(RAW_DIR, "banking_sector_2021_2025.csv")

NUMERIC_COLUMNS = [
    "PRICE HIGH (Rs.)",
    "PRICE LOW (Rs.)",
    "CLOSE PRICE (Rs.)",
    "OPEN PRICE (Rs.)",
    "TRADE VOLUME (No.)",
    "SHARE VOLUME (No.)",
    "TURNOVER (Rs.)",
]


def load_year_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop empty "Unnamed: N" trailer columns (present in the 2023 file).
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Strip whitespace from headers.
    df.columns = df.columns.str.strip()

    # Strip thousands separators / whitespace from numeric columns and cast
    # to float. Always round-trip through str: pandas may infer these
    # columns as a "string" extension dtype rather than plain "object"
    # (dtype == object would then miss them), and round-tripping already
    # numeric columns through str is harmless.
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )

    # Normalize the trading date to a proper datetime. Each yearly file uses
    # its own consistent format (day-first "DD-MON-YY" vs ISO
    # "YYYY-MM-DD HH:MM:SS"), so let pandas infer per-element rather than
    # forcing a single format/dayfirst rule across all files.
    df["TRADING DATE"] = pd.to_datetime(
        df["TRADING DATE"].astype(str).str.strip(), format="mixed", dayfirst=True
    )

    return df


def main() -> None:
    csv_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*_banking_sector.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No yearly banking sector CSVs found in {RAW_DIR}")

    print(f"Found {len(csv_paths)} files to merge:")
    for p in csv_paths:
        print(f"  - {os.path.basename(p)}")

    frames = [load_year_file(p) for p in csv_paths]
    merged = pd.concat(frames, ignore_index=True)

    # Drop exact duplicate rows that can arise from overlapping exports.
    merged = merged.drop_duplicates()

    merged = merged.sort_values(["COMPANY ID", "TRADING DATE"]).reset_index(drop=True)

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nMerged {len(merged)} rows -> {OUTPUT_PATH}")
    print(f"Date range: {merged['TRADING DATE'].min().date()} to {merged['TRADING DATE'].max().date()}")
    print(f"Companies: {merged['COMPANY ID'].nunique()}")


if __name__ == "__main__":
    main()
