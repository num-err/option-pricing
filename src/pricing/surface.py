"""Volatility smile and surface construction from a quoted option chain.

Pure data-transform functions: given an option chain (a pandas DataFrame
with columns describing strike, bid, ask, volume, and time to expiry),
filter out quotes too illiquid to trust and back out per-strike implied
volatility. No I/O or plotting lives here -- pulling data is
scripts/fetch_chain.py and rendering the smile is scripts/plot_smile.py.

Liquidity filter
------------------
A quote is dropped if either:
- it traded zero volume that day (a stale or uncrossed quote with no recent
  trade to corroborate the price), or
- its bid-ask spread relative to the mid price exceeds `max_spread_frac`
  (a wide relative spread means the quoted "price" is not pinned down well
  enough to trust an implied vol backed out of it).
The mid price ((bid + ask) / 2) is used as the option price for implied
vol, since it is a less biased estimate of fair value than either the bid
or the ask alone.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from . import implied_vol as iv

OptionType = Literal["call", "put"]


def log_moneyness(S: float, K: pd.Series) -> pd.Series:
    """log(K / S): 0 at the money, negative when K < S, positive when K > S."""
    return np.log(K / S)


def filter_liquid(chain: pd.DataFrame, max_spread_frac: float = 0.15) -> pd.DataFrame:
    """Drop quotes with zero traded volume or too wide a relative bid-ask spread.

    Expects "bid", "ask", "volume" columns. Returns a copy; does not mutate
    the input.
    """
    mid = (chain["bid"] + chain["ask"]) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_frac = (chain["ask"] - chain["bid"]) / mid
    liquid = (chain["volume"] > 0) & (mid > 0) & (spread_frac <= max_spread_frac)
    return chain.loc[liquid].copy()


def add_implied_vol(
    chain: pd.DataFrame,
    S: float,
    r: float,
    q: float,
    option_type: OptionType,
) -> pd.DataFrame:
    """Add "mid", "log_moneyness", and "implied_vol" columns to a chain.

    Expects "strike" and "T" (year fraction to expiry) columns, plus "bid"
    and "ask". Rows whose mid price violates the no-arbitrage bounds (e.g.
    a stale quote crossed by a recent move in the underlying) are dropped
    rather than left as NaN.
    """
    chain = chain.copy()
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2.0
    chain["log_moneyness"] = log_moneyness(S, chain["strike"])

    implied_vols = []
    for _, row in chain.iterrows():
        try:
            implied_vols.append(
                iv.implied_vol(row["mid"], S, row["strike"], row["T"], r, q, option_type)
            )
        except ValueError:
            implied_vols.append(np.nan)
    chain["implied_vol"] = implied_vols

    return chain.dropna(subset=["implied_vol"]).reset_index(drop=True)
