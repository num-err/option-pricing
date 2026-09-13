"""Closed-form Black-Scholes-Merton prices and analytic Greeks.

Model assumptions
------------------
- The underlying follows geometric Brownian motion under the risk-neutral
  measure with constant volatility: dS_t = (r - q) S_t dt + sigma S_t dW_t.
- The risk-free rate r and the continuous dividend yield q are constant over
  the life of the option.
- Volatility sigma is constant and known in advance.
- European exercise: the option can only be exercised at maturity T.
- Markets are frictionless: no transaction costs or taxes, unlimited
  borrowing/lending at rate r, unlimited short selling, continuous trading.

Every function is vectorized with numpy: S, K, T, r, q, sigma may be scalars
or arrays that broadcast together against each other. `option_type` selects
call or put for a single call and is not itself vectorized.

Degenerate inputs
------------------
The standard d1/d2 formula divides by `sigma * sqrt(T)`, which is undefined
at T = 0 or sigma = 0. Both are handled explicitly rather than left to
propagate NaN/inf:

- T == 0 (expired): the price is the intrinsic value, and delta is the
  0/1 (or 0/-1) indicator of moneyness. gamma, vega, theta, and rho are
  returned as 0 by convention -- there is no remaining time for volatility,
  vol-of-vol, time decay, or rate sensitivity to act through.
- sigma == 0 with T > 0 (deterministic): there is no randomness, so the
  terminal price S_T = S * exp((r - q) * T) is known today and the option
  price is that forward payoff discounted at r. Delta is exp(-qT) or
  -exp(-qT) when the deterministic payoff is in the money, else 0 (this is
  the exact limit of the usual delta formula). gamma, vega, theta, and rho
  are returned as 0 by convention for the same reason as above -- these are
  measures of sensitivity to randomness or to volatility that no longer
  exists at sigma = 0, and the formulas are ill-defined pointwise there.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _as_float_arrays(*args: ArrayLike) -> list[NDArray[np.float64]]:
    return [np.asarray(a, dtype=np.float64) for a in args]


def _validate_option_type(option_type: OptionType) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _d1_d2(
    S: NDArray[np.float64],
    K: NDArray[np.float64],
    T: NDArray[np.float64],
    r: NDArray[np.float64],
    q: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """d1 and d2 from the Black-Scholes formula.

    Only meaningful where T > 0 and sigma > 0. At degenerate points this
    divides by zero and returns inf/nan; callers must mask those entries out
    with their own degenerate-case handling rather than use these values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_sqrt_t = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
    return d1, d2


def _intrinsic(
    S: NDArray[np.float64], K: NDArray[np.float64], option_type: OptionType
) -> NDArray[np.float64]:
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


def _degenerate_masks(
    T: NDArray[np.float64], sigma: NDArray[np.float64]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Partition inputs into (expired, deterministic, normal) masks."""
    expired = T <= 0
    deterministic = (~expired) & (sigma <= 0)
    normal = ~(expired | deterministic)
    return expired, deterministic, normal


def _broadcast_inputs(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> list[NDArray[np.float64]]:
    arrays = _as_float_arrays(S, K, T, r, q, sigma)
    return list(np.broadcast_arrays(*arrays))


def price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
) -> NDArray[np.float64]:
    """European option price under Black-Scholes-Merton.

    Parameters
    ----------
    S : spot price of the underlying.
    K : strike price.
    T : time to maturity in years (T >= 0).
    r : continuously-compounded risk-free rate.
    q : continuous dividend yield.
    sigma : volatility of the underlying (sigma >= 0).
    option_type : "call" or "put".
    """
    _validate_option_type(option_type)
    S, K, T, r, q, sigma = _broadcast_inputs(S, K, T, r, q, sigma)
    expired, deterministic, _ = _degenerate_masks(T, sigma)

    intrinsic = _intrinsic(S, K, option_type)

    forward = S * np.exp((r - q) * T)
    disc_forward_payoff = np.exp(-r * T) * _intrinsic(forward, K, option_type)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        bs_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return np.where(expired, intrinsic, np.where(deterministic, disc_forward_payoff, bs_price))


def delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
) -> NDArray[np.float64]:
    """Sensitivity of price to the spot price, dV/dS."""
    _validate_option_type(option_type)
    S, K, T, r, q, sigma = _broadcast_inputs(S, K, T, r, q, sigma)
    expired, deterministic, _ = _degenerate_masks(T, sigma)

    forward = S * np.exp((r - q) * T)
    if option_type == "call":
        expired_delta = np.where(S > K, 1.0, 0.0)
        deterministic_delta = np.where(forward > K, np.exp(-q * T), 0.0)
    else:
        expired_delta = np.where(S < K, -1.0, 0.0)
        deterministic_delta = np.where(forward < K, -np.exp(-q * T), 0.0)

    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        bs_delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        bs_delta = -np.exp(-q * T) * norm.cdf(-d1)

    return np.where(expired, expired_delta, np.where(deterministic, deterministic_delta, bs_delta))


def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> NDArray[np.float64]:
    """Curvature of price with respect to spot, d^2V/dS^2. Identical for calls and puts."""
    S, K, T, r, q, sigma = _broadcast_inputs(S, K, T, r, q, sigma)
    expired, deterministic, _ = _degenerate_masks(T, sigma)

    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        bs_gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return np.where(expired | deterministic, 0.0, bs_gamma)


def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> NDArray[np.float64]:
    """Sensitivity of price to volatility, dV/dsigma. Identical for calls and puts."""
    S, K, T, r, q, sigma = _broadcast_inputs(S, K, T, r, q, sigma)
    expired, deterministic, _ = _degenerate_masks(T, sigma)

    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        bs_vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    return np.where(expired | deterministic, 0.0, bs_vega)


def theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
) -> NDArray[np.float64]:
    """Sensitivity of price to the passage of calendar time, dV/dt = -dV/dT."""
    _validate_option_type(option_type)
    S, K, T, r, q, sigma = _broadcast_inputs(S, K, T, r, q, sigma)
    expired, deterministic, _ = _degenerate_masks(T, sigma)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay_term = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        bs_theta = (
            decay_term + q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
    else:
        bs_theta = (
            decay_term - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )

    return np.where(expired | deterministic, 0.0, bs_theta)


def rho(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
) -> NDArray[np.float64]:
    """Sensitivity of price to the risk-free rate, dV/dr."""
    _validate_option_type(option_type)
    S, K, T, r, q, sigma = _broadcast_inputs(S, K, T, r, q, sigma)
    expired, deterministic, _ = _degenerate_masks(T, sigma)

    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        bs_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        bs_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return np.where(expired | deterministic, 0.0, bs_rho)
