"""Back out implied vol from the cached SPY chain and plot the smile.

Reads data/spy_chain.csv (written by fetch_chain.py) so this is
reproducible offline. For each of the two cached expiries:

- Filters illiquid quotes (pricing.surface.filter_liquid: zero volume, or a
  bid-ask spread wider than 15% of the mid).
- Keeps only out-of-the-money options (puts for strike <= spot, calls for
  strike >= spot) and drops the rest. This is standard practice for
  building an equity-index smile: OTM options trade far more volume than
  their ITM counterparts at the same strike (traders use them for
  directional bets and hedges), so their quotes are tighter and more
  trustworthy, and splicing OTM puts with OTM calls at the money gives one
  continuous curve across the whole strike range instead of two noisier,
  overlapping ones.
- Backs out implied vol per strike (pricing.surface.add_implied_vol), which
  internally drops any quote that violates the no-arbitrage price bounds.

Saves figures/smile.png: implied vol vs. log-moneyness log(K/S), one curve
per expiry.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pricing import surface

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spy_chain.csv"
FIGURE_PATH = Path(__file__).resolve().parent.parent / "figures" / "smile.png"
MAX_SPREAD_FRAC = 0.15


def build_expiry_smile(expiry_chain: pd.DataFrame, S: float, r: float, q: float) -> pd.DataFrame:
    liquid = surface.filter_liquid(expiry_chain, max_spread_frac=MAX_SPREAD_FRAC)

    otm_puts = liquid[(liquid["option_type"] == "put") & (liquid["strike"] <= S)]
    otm_calls = liquid[(liquid["option_type"] == "call") & (liquid["strike"] >= S)]

    smile_puts = surface.add_implied_vol(otm_puts, S, r, q, "put")
    smile_calls = surface.add_implied_vol(otm_calls, S, r, q, "call")

    return pd.concat([smile_puts, smile_calls], ignore_index=True).sort_values("log_moneyness")


def main() -> None:
    chain = pd.read_csv(DATA_PATH)
    S = float(chain["S"].iloc[0])
    r = float(chain["r"].iloc[0])
    q = float(chain["q"].iloc[0])

    expiries = sorted(chain["expiry"].unique(), key=lambda e: chain.loc[chain["expiry"] == e, "T"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for expiry in expiries:
        expiry_chain = chain[chain["expiry"] == expiry]
        T = float(expiry_chain["T"].iloc[0])
        smile = build_expiry_smile(expiry_chain, S, r, q)
        if smile.empty:
            print(f"expiry {expiry}: no strikes survived filtering, skipping")
            continue
        label = f"{expiry} (T={T * 365:.1f}d, n={len(smile)})"
        ax.plot(smile["log_moneyness"], smile["implied_vol"], marker="o", markersize=4, label=label)

    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("log-moneyness  log(K / S)")
    ax.set_ylabel("Black-Scholes implied volatility")
    ax.set_title(f"SPY implied volatility smile  (S={S:.2f}, fetched from cached chain)")
    ax.legend()
    fig.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
