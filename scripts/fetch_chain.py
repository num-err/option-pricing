"""Pull a live SPY option chain via yfinance and cache it to data/ as CSV.

This script is the only place in the project that touches the network. It
is run once to produce data/spy_chain.csv, which is committed so that
scripts/plot_smile.py (and anyone re-running this project) can reproduce the
README figure offline without depending on market hours or a live feed.

Assumptions documented here rather than derived:
- Risk-free rate r: the most recent close of ^IRX (13-week T-bill discount
  yield), a standard short-rate proxy for pricing near-dated equity options.
- Dividend yield q: yfinance's trailingAnnualDividendYield for SPY (its
  `dividendYield` field is inconsistently scaled across yfinance versions,
  so it is deliberately not used).
- Time to expiry T: calendar days between now and each expiry, divided by
  365.0. This ignores the trading-day-vs-calendar-day distinction some desks
  use; it is a common simplification and does not materially change the
  smile shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "SPY"
N_EXPIRIES = 2
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "spy_chain.csv"

CHAIN_COLUMNS = ["strike", "bid", "ask", "volume", "openInterest"]


def fetch_risk_free_rate() -> float:
    irx = yf.Ticker("^IRX").history(period="5d")
    return float(irx["Close"].iloc[-1]) / 100.0


def fetch_dividend_yield(ticker: yf.Ticker) -> float:
    return float(ticker.get_info().get("trailingAnnualDividendYield", 0.013))


def fetch_spot(ticker: yf.Ticker) -> float:
    return float(ticker.fast_info["lastPrice"])


def main() -> None:
    ticker = yf.Ticker(TICKER)
    S = fetch_spot(ticker)
    r = fetch_risk_free_rate()
    q = fetch_dividend_yield(ticker)
    now = datetime.now(timezone.utc)

    expiries = ticker.options[:N_EXPIRIES]
    rows = []
    for expiry_str in expiries:
        expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(
            hour=21, minute=0, tzinfo=timezone.utc  # ~4pm ET close
        )
        T = (expiry_dt - now).total_seconds() / (365.0 * 24 * 3600)
        if T <= 0:
            continue

        chain = ticker.option_chain(expiry_str)
        for option_type, df in (("call", chain.calls), ("put", chain.puts)):
            subset = df[CHAIN_COLUMNS].copy()
            subset["option_type"] = option_type
            subset["expiry"] = expiry_str
            subset["T"] = T
            rows.append(subset)

    combined = pd.concat(rows, ignore_index=True)
    combined["S"] = S
    combined["r"] = r
    combined["q"] = q
    combined["fetched_at"] = now.isoformat()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(combined)} rows ({len(expiries)} expiries) to {OUTPUT_PATH}")
    print(f"S={S:.2f} r={r:.4f} q={q:.4f}")


if __name__ == "__main__":
    main()
