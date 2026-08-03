"""
Client for the CSE historical charts API.

Verified against a real browser-captured request (2026-08-02):
    POST https://www.cse.lk/api/charts
    Content-Type: application/x-www-form-urlencoded
    Cookie: accessToken=<token>
    Body: symbol=<TICKER>.N0000&fromDate=DD-MM-YYYY&toDate=DD-MM-YYYY&period=1

This is NOT a public/anonymous endpoint — it requires a live session's
accessToken (see config.get_cse_access_token()), and that token expires.
If requests start failing with CSEAuthError, capture a fresh token via
browser DevTools and drop it into backend/cse_token.txt.

Response field names are NOT YET CONFIRMED end-to-end — testing so far only
got as far as an expired-token response. fetch_symbol_history() parses
defensively and raises a clear, inspectable RuntimeError (showing the
actual keys it saw) if a row doesn't match FIELD_MAP, instead of silently
returning wrong data.
"""
import logging
from datetime import datetime
from typing import Dict, List

import requests

from .config import CSE_CHARTS_URL, get_cse_access_token

logger = logging.getLogger("backend.cse_client")

# Confirmed for COMB ("COMB.N0000"); ASSUMED identical for the other 9
# tickers — verify each once real data comes back, in case any of your 10
# banks uses a different share-class suffix (e.g. non-voting).
SYMBOL_SUFFIX = ".N0000"

FIELD_MAP = {
    "date": ["date", "tradeDate", "tradingDate", "x", "timestamp"],
    "open": ["open", "openPrice", "o"],
    "high": ["high", "highPrice", "h"],
    "low": ["low", "lowPrice", "l"],
    "close": ["close", "closePrice", "c", "price"],
    "volume": ["volume", "shareVolume", "tradeVolume", "v"],
}


class CSEAuthError(RuntimeError):
    """Raised when the access token is missing, expired, or rejected."""


def _first_present(row: dict, keys: List[str]):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def fetch_symbol_history(ticker: str, from_date: datetime, to_date: datetime) -> List[dict]:
    """
    Fetch daily OHLCV rows for one ticker between from_date and to_date
    (inclusive). Returns a list of {"date","open","high","low","close",
    "volume"} dicts. Raises CSEAuthError if the token is missing/expired,
    RuntimeError for any other unexpected response shape.
    """
    token = get_cse_access_token()
    if not token:
        raise CSEAuthError(
            "No CSE access token configured. Set the CSE_ACCESS_TOKEN "
            "environment variable, or paste it into backend/cse_token.txt."
        )

    symbol = f"{ticker.upper()}{SYMBOL_SUFFIX}"
    body = {
        "symbol": symbol,
        "fromDate": from_date.strftime("%d-%m-%Y"),
        "toDate": to_date.strftime("%d-%m-%Y"),
        "period": "1",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded",
        "origin": "https://www.cse.lk",
        "referer": f"https://www.cse.lk/company-profile?symbol={symbol}",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    try:
        response = requests.post(
            CSE_CHARTS_URL,
            data=body,
            headers=headers,
            cookies={"accessToken": token},
            timeout=15,
        )
    except Exception as exc:
        raise RuntimeError(f"CSE charts request failed for {symbol}: {exc}") from exc

    if response.status_code in (401, 403, 417):
        raise CSEAuthError(
            f"CSE rejected the access token for {symbol} (HTTP "
            f"{response.status_code}). It has likely expired — capture a "
            f"fresh one and update backend/cse_token.txt."
        )
    if not response.ok:
        raise RuntimeError(
            f"CSE charts request for {symbol} failed: HTTP "
            f"{response.status_code} — {response.text[:200]}"
        )

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"CSE charts response for {symbol} wasn't valid JSON: "
            f"{response.text[:200]}"
        ) from exc

    rows = payload if isinstance(payload, list) else payload.get("data", [])
    if not rows:
        raise RuntimeError(
            f"CSE charts returned no rows for {symbol} between "
            f"{body['fromDate']} and {body['toDate']} — either the token is "
            f"stale/rejected without an explicit 401/403/417, or there was "
            f"genuinely no trading in this range."
        )

    parsed = []
    for row in rows:
        date_val = _first_present(row, FIELD_MAP["date"])
        close_val = _first_present(row, FIELD_MAP["close"])
        if date_val is None or close_val is None:
            raise RuntimeError(
                f"CSE charts row for {symbol} didn't match the expected "
                f"field names — got keys {list(row.keys())}. Update "
                f"FIELD_MAP in backend/cse_client.py to match this real "
                f"response shape."
            )
        parsed.append(
            {
                "date": date_val,
                "open": _first_present(row, FIELD_MAP["open"]),
                "high": _first_present(row, FIELD_MAP["high"]),
                "low": _first_present(row, FIELD_MAP["low"]),
                "close": close_val,
                "volume": _first_present(row, FIELD_MAP["volume"]),
            }
        )

    return parsed


def fetch_all_history(
    tickers: List[str], from_date: datetime, to_date: datetime
) -> Dict[str, List[dict]]:
    """Fetch history for every ticker — one request per symbol, matching
    how the real site itself calls this endpoint (no bulk/multi-symbol mode
    was observed)."""
    return {ticker: fetch_symbol_history(ticker, from_date, to_date) for ticker in tickers}
