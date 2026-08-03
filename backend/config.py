"""
Backend configuration — paths, constants, and the CSE ticker -> display-name
mapping for the 10-asset banking-sector universe this model was trained on.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_backend_env() -> None:
    """Load backend/.env into os.environ if present."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_backend_env()
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# Must match src/config/config.py's DataConfig defaults — kept independent
# here so the backend doesn't need to import training-time config.
WINDOW_SIZE = 30
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
INITIAL_CAPITAL = 1_000_000.0

TEST_SUMMARY_PATH = os.path.join(RESULTS_DIR, "test_summary.json")
TEST_WEIGHTS_PATH = os.path.join(RESULTS_DIR, "test_weights.csv")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
MANIFEST_PATH = os.path.join(MODELS_DIR, "manifest.json")

# Ticker -> human-readable name, from the raw CSVs' SHORT NAME column.
ASSET_DISPLAY_NAMES = {
    "ABL": "Amana Bank",
    "COMB": "Commercial Bank",
    "DFCC": "DFCC Bank PLC",
    "HNB": "HNB",
    "NDB": "National Development Bank",
    "NTB": "Nations Trust Bank",
    "PABC": "Pan Asia Bank",
    "SAMP": "Sampath Bank",
    "SEYB": "Seylan Bank",
    "UBC": "Union Bank",
}

# Verified against a real browser-captured request (2026-08-02):
# POST, form-urlencoded body: symbol=<TICKER>.N0000&fromDate=DD-MM-YYYY&
# toDate=DD-MM-YYYY&period=1 — returns per-symbol historical daily rows.
# Requires a live session's accessToken cookie (see get_cse_access_token()
# below) — this is NOT a public/anonymous endpoint.
CSE_CHARTS_URL = "https://www.cse.lk/api/charts"

# The access token expires (session-based), so it's never hardcoded here.
# Set it either as an environment variable, or drop it in
# backend/cse_token.txt (gitignored — see .gitignore) — whichever is easier
# to refresh each time your browser session gives you a new one.
_TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cse_token.txt")


def get_cse_access_token():
    """Returns the current CSE accessToken, or None if not configured.

    Accepts either a raw token in backend/cse_token.txt, or a
    `CSE_ACCESS_TOKEN=<token>` line (same for the environment variable)."""
    env_token = os.environ.get("CSE_ACCESS_TOKEN") or os.environ.get("CSE_ACCESS_TOKE")
    if env_token:
        preview = env_token[:20] + "..." + env_token[-8:] if len(env_token) > 30 else env_token
        print(f"env_token loaded: {preview}")
        return _strip_prefix(env_token)

    print("env_token: None")

    if os.path.exists(_TOKEN_FILE):
        with open(_TOKEN_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return _strip_prefix(content) if content else None

    return None


def _strip_prefix(value: str) -> str:
    value = value.strip()
    if value.upper().startswith("CSE_ACCESS_TOKEN="):
        value = value.split("=", 1)[1]
    return value.strip()
