"""Cox-Ross-Rubinstein binomial tree pricing with optional American exercise.

Model assumptions
------------------
- Approximates the same risk-neutral GBM dynamics as Black-Scholes-Merton
  (dS = (r - q) S dt + sigma S dW) with a discrete-time, recombining binomial
  lattice: over each step of length dt = T/N the underlying moves up by a
  factor u = exp(sigma*sqrt(dt)) or down by d = 1/u.
- The risk-neutral up probability
      p = (exp((r - q) * dt) - d) / (u - d)
  is the unique value making the discounted, dividend-reinvested underlying
  a martingale under the tree measure. As N -> infinity the tree's terminal
  distribution converges in distribution to the same lognormal used by
  Black-Scholes (Cox, Ross & Rubinstein 1979), so the European tree price
  converges to the closed-form price.
- r, q, and sigma are constant over the life of the option.
- American exercise (when enabled) is checked at every node by backward
  induction: node value = max(discounted continuation value, intrinsic
  value). This is the standard optimal-stopping / dynamic-programming
  argument, applied exactly rather than approximated.

Only scalar S, K, T, r, q, sigma are supported (one option per call); within
a call, each time step of the backward induction is vectorized across the
nodes alive at that step with numpy.

Known limitation: sigma = 0 is not supported. Unlike the closed-form model
(black_scholes.price), whose sigma = 0 limit is well-defined because the
formula separates drift (S * exp((r-q)T)) from volatility, the CRR tree
encodes volatility directly into the node spacing (u = exp(sigma*sqrt(dt))).
At sigma = 0, u = d = 1: every node collapses to the same spot price S, so
the tree cannot represent the deterministic forward drift at all. Rather
than silently return the wrong number, this raises ValueError.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

OptionType = Literal["call", "put"]


def _intrinsic(S: NDArray[np.float64], K: float, option_type: OptionType) -> NDArray[np.float64]:
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N: int,
    option_type: OptionType = "call",
    american: bool = False,
) -> float:
    """Price a European or American option on an N-step CRR binomial tree.

    Parameters
    ----------
    S, K : spot price and strike (scalars).
    T : time to maturity in years (T >= 0).
    r : continuously-compounded risk-free rate.
    q : continuous dividend yield.
    sigma : volatility of the underlying (sigma > 0; see module docstring).
    N : number of time steps in the tree (N >= 1).
    option_type : "call" or "put".
    american : if True, allow early exercise at every node; if False, price
        the European (exercise-only-at-maturity) option.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if sigma <= 0:
        raise ValueError("binomial.price requires sigma > 0 (see module docstring)")
    if T <= 0:
        return float(_intrinsic(np.array(S), K, option_type))

    dt = T / N
    x = sigma * np.sqrt(dt)  # log up-move size: u = exp(x), d = exp(-x)
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - np.exp(-x)) / (np.exp(x) - np.exp(-x))
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"risk-neutral probability p={p:.4f} outside (0, 1) for these parameters; "
            "the tree step is not arbitrage-free (try a larger N)."
        )

    # Terminal spot prices: j up-moves and (N - j) down-moves, j = 0..N.
    # Written as S * exp(x * (2j - N)) rather than S * u**j * d**(N-j) to
    # avoid accumulating floating-point error from repeated powers of u, d.
    j = np.arange(N + 1)
    S_terminal = S * np.exp(x * (2 * j - N))
    values = _intrinsic(S_terminal, K, option_type)

    for step in range(N - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            j = np.arange(step + 1)
            S_step = S * np.exp(x * (2 * j - step))
            values = np.maximum(values, _intrinsic(S_step, K, option_type))

    return float(values[0])
