"""Tests for pricing.black_scholes: analytic Greeks against finite differences
of the price function, monotonicity, moneyness limits, and degenerate inputs.
"""

import itertools

import numpy as np
import pytest

from pricing import black_scholes as bs

# A grid of "well-behaved" parameters: T > 0 and sigma > 0 everywhere, since
# finite differences of the price function are not meaningful exactly at the
# degenerate points (T = 0, sigma = 0) that black_scholes.py handles as
# special cases.
S_GRID = [60.0, 90.0, 100.0, 110.0, 150.0]
K_GRID = [80.0, 100.0, 120.0]
T_GRID = [0.25, 1.0, 2.5]
R_GRID = [0.0, 0.03, 0.07]
Q_GRID = [0.0, 0.02]
SIGMA_GRID = [0.1, 0.3, 0.6]

GRID = list(itertools.product(S_GRID, K_GRID, T_GRID, R_GRID, Q_GRID, SIGMA_GRID))


def _params():
    S, K, T, r, q, sigma = (np.array(x) for x in zip(*GRID))
    return S, K, T, r, q, sigma


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(option_type):
    S, K, T, r, q, sigma = _params()
    h = 1e-4
    fd = (
        bs.price(S + h, K, T, r, q, sigma, option_type)
        - bs.price(S - h, K, T, r, q, sigma, option_type)
    ) / (2 * h)
    analytic = bs.delta(S, K, T, r, q, sigma, option_type)
    np.testing.assert_allclose(analytic, fd, atol=1e-5)


def test_gamma_matches_finite_difference():
    S, K, T, r, q, sigma = _params()
    h = 1e-2
    # Gamma is identical for calls and puts, so either suffices.
    fd = (
        bs.price(S + h, K, T, r, q, sigma, "call")
        - 2 * bs.price(S, K, T, r, q, sigma, "call")
        + bs.price(S - h, K, T, r, q, sigma, "call")
    ) / h**2
    analytic = bs.gamma(S, K, T, r, q, sigma)
    np.testing.assert_allclose(analytic, fd, atol=1e-5)


def test_vega_matches_finite_difference():
    S, K, T, r, q, sigma = _params()
    h = 1e-5
    fd = (
        bs.price(S, K, T, r, q, sigma + h, "call")
        - bs.price(S, K, T, r, q, sigma - h, "call")
    ) / (2 * h)
    analytic = bs.vega(S, K, T, r, q, sigma)
    np.testing.assert_allclose(analytic, fd, atol=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_finite_difference(option_type):
    S, K, T, r, q, sigma = _params()
    h = 1e-5
    # theta = dV/dt = -dV/dT: price is parameterized by time-to-maturity T,
    # which shrinks as calendar time passes, so the sign flips.
    dV_dT = (
        bs.price(S, K, T + h, r, q, sigma, option_type)
        - bs.price(S, K, T - h, r, q, sigma, option_type)
    ) / (2 * h)
    fd = -dV_dT
    analytic = bs.theta(S, K, T, r, q, sigma, option_type)
    np.testing.assert_allclose(analytic, fd, atol=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(option_type):
    S, K, T, r, q, sigma = _params()
    h = 1e-5
    fd = (
        bs.price(S, K, T, r + h, q, sigma, option_type)
        - bs.price(S, K, T, r - h, q, sigma, option_type)
    ) / (2 * h)
    analytic = bs.rho(S, K, T, r, q, sigma, option_type)
    np.testing.assert_allclose(analytic, fd, atol=1e-5)


def test_call_price_increasing_in_spot():
    S = np.linspace(50.0, 200.0, 50)
    for K, T, r, q, sigma in itertools.product(K_GRID, T_GRID, R_GRID, Q_GRID, SIGMA_GRID):
        prices = bs.price(S, K, T, r, q, sigma, "call")
        assert np.all(np.diff(prices) > 0)


def test_call_price_increasing_in_volatility():
    # Vega is strictly positive for every sigma > 0, so the true price curve
    # is strictly increasing. Deep in/out of the money combined with very low
    # sigma pushes consecutive prices below float64 resolution (vega ~ 0 to
    # machine precision there), so we check non-decreasing step-to-step and
    # a strict increase end-to-end rather than every single step.
    sigma = np.linspace(0.05, 1.5, 50)
    for S, K, T, r, q in itertools.product([80.0, 100.0, 120.0], K_GRID, T_GRID, R_GRID, Q_GRID):
        prices = bs.price(S, K, T, r, q, sigma, "call")
        assert np.all(np.diff(prices) >= 0)
        assert prices[-1] > prices[0]


def test_deep_itm_call_delta_approaches_discounted_dividend_factor():
    S, K, T, r, q, sigma = 1.0e6, 100.0, 1.0, 0.03, 0.02, 0.2
    d = bs.delta(S, K, T, r, q, sigma, "call")
    assert d == pytest.approx(np.exp(-q * T), abs=1e-8)


def test_deep_otm_call_delta_approaches_zero():
    S, K, T, r, q, sigma = 1.0e-3, 100.0, 1.0, 0.03, 0.02, 0.2
    d = bs.delta(S, K, T, r, q, sigma, "call")
    assert d == pytest.approx(0.0, abs=1e-8)


class TestDegenerateInputs:
    """T = 0 and sigma = 0 are special-cased in black_scholes.py; check the
    documented behavior directly rather than via finite differences, which
    are not meaningful at these boundary points."""

    def test_expired_call_is_intrinsic_value(self):
        S, K = np.array([80.0, 100.0, 120.0]), 100.0
        p = bs.price(S, K, 0.0, 0.05, 0.02, 0.2, "call")
        np.testing.assert_allclose(p, np.maximum(S - K, 0.0))

    def test_expired_put_is_intrinsic_value(self):
        S, K = np.array([80.0, 100.0, 120.0]), 100.0
        p = bs.price(S, K, 0.0, 0.05, 0.02, 0.2, "put")
        np.testing.assert_allclose(p, np.maximum(K - S, 0.0))

    def test_zero_vol_call_is_discounted_deterministic_payoff(self):
        S, K, T, r, q = 110.0, 100.0, 1.0, 0.05, 0.01
        p = bs.price(S, K, T, r, q, 0.0, "call")
        forward = S * np.exp((r - q) * T)
        expected = np.exp(-r * T) * max(forward - K, 0.0)
        assert p == pytest.approx(expected)

    def test_zero_vol_call_delta_is_indicator_scaled_by_dividend_discount(self):
        S, K, T, r, q = 110.0, 100.0, 1.0, 0.05, 0.01
        d = bs.delta(S, K, T, r, q, 0.0, "call")
        assert d == pytest.approx(np.exp(-q * T))

        S_otm = 90.0
        d_otm = bs.delta(S_otm, K, T, r, q, 0.0, "call")
        assert d_otm == pytest.approx(0.0)
