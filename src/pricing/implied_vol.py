"""Implied volatility recovery via Brent's method.

Given a quoted price, `implied_vol` finds the sigma that makes
`black_scholes.price(..., sigma)` equal to that price. This is a 1-D
root-find because, under the Black-Scholes-Merton assumptions, price is a
strictly increasing, continuous function of sigma for any T > 0 -- from a
lower bound (the sigma -> 0 deterministic price) up to an upper bound
(the sigma -> infinity price) -- so exactly one sigma reproduces any price
strictly between those bounds. Brent's method (`scipy.optimize.brentq`)
exploits that monotonicity: it only needs a bracket where the objective
changes sign, not a derivative, and converges superlinearly.

No-arbitrage bounds
--------------------
Those same lower/upper limits are model-free no-arbitrage bounds, not just
artifacts of the search bracket:
    call: max(S*exp(-qT) - K*exp(-rT), 0)  <=  C  <=  S*exp(-qT)
    put:  max(K*exp(-rT) - S*exp(-qT), 0)  <=  P  <=  K*exp(-rT)
A quoted price outside these bounds cannot correspond to any volatility --
not "a very large or very small one," but genuinely none, because it would
imply an arbitrage against a static replicating portfolio (e.g. a call
priced above S*exp(-qT) is priced above owning the stock outright with
unlimited upside). Silently returning NaN or an extrapolated number for
such an input would hide a bad quote; this module raises instead.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import brentq

from . import black_scholes as bs

OptionType = Literal["call", "put"]

_DEFAULT_SIGMA_BRACKET = (1e-6, 5.0)


class ArbitrageViolation(ValueError):
    """A quoted price falls outside the model-free no-arbitrage bounds, so no
    volatility (however large or small) could have produced it."""


def price_bounds(
    S: float, K: float, T: float, r: float, q: float, option_type: OptionType
) -> tuple[float, float]:
    """Model-free no-arbitrage bounds (lower, upper) for a European price."""
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option_type == "call":
        return max(disc_S - disc_K, 0.0), disc_S
    return max(disc_K - disc_S, 0.0), disc_K


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType = "call",
    sigma_bracket: tuple[float, float] = _DEFAULT_SIGMA_BRACKET,
) -> float:
    """Invert black_scholes.price for the volatility consistent with `price`.

    Parameters
    ----------
    price : quoted option price.
    S, K, T, r, q : as in black_scholes.price; T must be > 0.
    option_type : "call" or "put".
    sigma_bracket : (low, high) search interval for Brent's method. The
        default (1e-6, 5.0) comfortably covers realistic equity/index
        implied vols; pass a wider bracket for instruments with extreme vol.

    Raises
    ------
    ArbitrageViolation : `price` is outside the no-arbitrage bounds for
        these parameters -- no sigma could reproduce it.
    ValueError : T <= 0 (implied vol is undefined once there is no time
        value to invert) or option_type is invalid.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if T <= 0:
        raise ValueError("implied_vol is undefined for T <= 0 (no time value to invert)")

    lower, upper = price_bounds(S, K, T, r, q, option_type)
    tol = 1e-10 * max(1.0, upper)
    if price < lower - tol or price > upper + tol:
        raise ArbitrageViolation(
            f"price={price:.6f} violates no-arbitrage bounds "
            f"[{lower:.6f}, {upper:.6f}] for this {option_type} "
            f"(S={S}, K={K}, T={T}, r={r}, q={q})"
        )

    def objective(sigma: float) -> float:
        return float(bs.price(S, K, T, r, q, sigma, option_type)) - price

    lo, hi = sigma_bracket
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo > 0 or f_hi < 0:
        raise ArbitrageViolation(
            f"price={price:.6f} is within the theoretical no-arbitrage bounds "
            f"[{lower:.6f}, {upper:.6f}] but outside the price range spanned by "
            f"sigma_bracket={sigma_bracket} for this {option_type}; pass a wider "
            "sigma_bracket."
        )

    return brentq(objective, lo, hi, xtol=1e-12, rtol=1e-12)
