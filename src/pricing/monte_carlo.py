"""Monte Carlo pricing of European payoffs under GBM, with variance reduction.

Model assumptions
------------------
- Same risk-neutral dynamics as black_scholes and binomial: under the
  risk-neutral measure the underlying follows GBM,
      dS_t = (r - q) S_t dt + sigma S_t dW_t,
  so the terminal price has the closed-form solution
      S_T = S * exp((r - q - 0.5*sigma^2) T + sigma*sqrt(T) Z),  Z ~ N(0, 1).
  Because only the terminal value is needed for a European payoff, paths are
  sampled directly at maturity rather than stepped through time -- there is
  no discretization error from this, unlike path-dependent payoffs.
- r, q, and sigma are constant over [0, T].
- The option is European: the payoff depends only on S_T.

Every estimator returns an `MCResult(price, std_error)`: the payoff mean
discounted at r, and the standard error of that mean (sample std / sqrt(n)),
which is what makes "within k standard errors of the true price" a
meaningful statement about estimator accuracy rather than a guess.

Randomness uses numpy's Generator API (`np.random.default_rng`) rather than
the legacy global RandomState, so a simulation is reproducible from an
explicitly passed `rng` without mutating global state that other code (or
other tests) might depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo price estimate and its standard error."""

    price: float
    std_error: float


def _intrinsic(S: NDArray[np.float64], K: float, option_type: OptionType) -> NDArray[np.float64]:
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


def _validate(option_type: OptionType, n_paths: int) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if n_paths < 2:
        raise ValueError(f"n_paths must be >= 2 to estimate a standard error, got {n_paths}")


def _mean_and_se(discounted_values: NDArray[np.float64]) -> MCResult:
    n = discounted_values.shape[0]
    return MCResult(
        price=float(discounted_values.mean()),
        std_error=float(discounted_values.std(ddof=1) / np.sqrt(n)),
    )


def price_naive(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    option_type: OptionType,
    rng: np.random.Generator,
) -> MCResult:
    """Naive Monte Carlo: simulate n_paths independent terminal prices."""
    _validate(option_type, n_paths)
    Z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    discounted = np.exp(-r * T) * _intrinsic(S_T, K, option_type)
    return _mean_and_se(discounted)


def price_antithetic(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    option_type: OptionType,
    rng: np.random.Generator,
) -> MCResult:
    """Antithetic variates: pair each draw Z with -Z and average the payoffs.

    Reduces variance because a payoff that is monotone in Z (true of vanilla
    calls and puts) has negatively correlated values at Z and -Z, so the pair
    average has lower variance than two independent draws at the same total
    path count. Uses n_paths // 2 pairs (n_paths total simulated draws) so
    the compute budget matches price_naive(n_paths=...) exactly.
    """
    _validate(option_type, n_paths)
    n_pairs = n_paths // 2
    if n_pairs < 1:
        raise ValueError(f"n_paths must be >= 2 to form at least one antithetic pair, got {n_paths}")
    Z = rng.standard_normal(n_pairs)
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T_up = S * np.exp(drift + diffusion)
    S_T_down = S * np.exp(drift - diffusion)
    pair_payoff = 0.5 * (
        _intrinsic(S_T_up, K, option_type) + _intrinsic(S_T_down, K, option_type)
    )
    discounted = np.exp(-r * T) * pair_payoff
    return _mean_and_se(discounted)


def price_control_variate(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    option_type: OptionType,
    rng: np.random.Generator,
) -> MCResult:
    """Control variate: the discounted terminal stock price, whose expectation
    E[exp(-rT) S_T] = S*exp(-qT) is known in closed form.

    For each path, adjust the discounted payoff Y by the pathwise error of
    the control X = exp(-rT) S_T against its known mean:
        Y_adjusted = Y - b * (X - E[X]),   b = Cov(Y, X) / Var(X).
    b is the value minimizing Var(Y_adjusted), estimated from the same
    sample (a standard, asymptotically negligible bias for large n_paths).
    Y_adjusted has the same expectation as Y but lower variance whenever Y
    and X are correlated -- true here because both are increasing functions
    of the same S_T.
    """
    _validate(option_type, n_paths)
    Z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    Y = np.exp(-r * T) * _intrinsic(S_T, K, option_type)
    X = np.exp(-r * T) * S_T
    E_X = S * np.exp(-q * T)

    cov_matrix = np.cov(Y, X, ddof=1)
    b = cov_matrix[0, 1] / cov_matrix[1, 1]
    adjusted = Y - b * (X - E_X)
    return _mean_and_se(adjusted)
