"""Tests for pricing.implied_vol: round-trip recovery of a known volatility,
and that arbitrage-violating prices raise rather than returning garbage.
"""

import itertools

import numpy as np
import pytest

from pricing import black_scholes as bs
from pricing import implied_vol as iv

S_GRID = [80.0, 100.0, 120.0]
K_GRID = [80.0, 100.0, 120.0]
T_GRID = [0.1, 1.0, 2.0]
R_GRID = [0.0, 0.03, 0.08]
Q_GRID = [0.0, 0.02]
SIGMA_GRID = [0.05, 0.2, 0.5, 1.2]


# Below this vega, recovering sigma to 1e-6 is not numerically meaningful:
# deep ITM/OTM, short-dated, low-vol combinations make price essentially flat
# in sigma (vega underflows towards 0), so many different sigmas reproduce
# the same float64 price. That is a property of the inversion problem itself
# (real-world IV solvers are known to be unreliable in this regime), not a
# bug in implied_vol -- so the round-trip check is restricted to inputs
# where the price is actually sensitive to sigma.
MIN_VEGA_FOR_STABLE_INVERSION = 1e-2


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_round_trip_recovers_known_sigma(option_type):
    combos = itertools.product(S_GRID, K_GRID, T_GRID, R_GRID, Q_GRID, SIGMA_GRID)
    checked = 0
    for S, K, T, r, q, sigma_true in combos:
        if float(bs.vega(S, K, T, r, q, sigma_true)) < MIN_VEGA_FOR_STABLE_INVERSION:
            continue
        price = float(bs.price(S, K, T, r, q, sigma_true, option_type))
        recovered = iv.implied_vol(price, S, K, T, r, q, option_type)
        assert recovered == pytest.approx(sigma_true, abs=1e-6), (
            f"S={S} K={K} T={T} r={r} q={q} sigma={sigma_true} option={option_type}"
        )
        checked += 1
    assert checked > 400  # sanity check that the vega filter isn't vacuous


def test_price_above_upper_bound_raises():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    _, upper = iv.price_bounds(S, K, T, r, q, "call")
    with pytest.raises(iv.ArbitrageViolation):
        iv.implied_vol(upper + 1.0, S, K, T, r, q, "call")


def test_price_below_lower_bound_raises():
    S, K, T, r, q = 100.0, 120.0, 1.0, 0.05, 0.0
    lower, _ = iv.price_bounds(S, K, T, r, q, "call")
    assert lower == pytest.approx(0.0)
    with pytest.raises(iv.ArbitrageViolation):
        iv.implied_vol(-1.0, S, K, T, r, q, "call")


def test_negative_put_price_raises():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    with pytest.raises(iv.ArbitrageViolation):
        iv.implied_vol(-0.5, S, K, T, r, q, "put")


def test_arbitrage_violation_is_a_value_error():
    # Callers that only expect ValueError (e.g. from a generic invalid-input
    # path) should still catch this without needing to know the subclass.
    assert issubclass(iv.ArbitrageViolation, ValueError)


def test_expired_option_raises():
    with pytest.raises(ValueError):
        iv.implied_vol(5.0, 100.0, 100.0, 0.0, 0.05, 0.0, "call")


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        iv.implied_vol(5.0, 100.0, 100.0, 1.0, 0.05, 0.0, "straddle")
