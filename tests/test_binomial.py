"""Tests for pricing.binomial: convergence to Black-Scholes, American vs
European exercise, and the "no early exercise for a call with q=0" result.
"""

import numpy as np
import pytest

from pricing import binomial as bn
from pricing import black_scholes as bs


def test_european_tree_converges_to_black_scholes():
    # At-the-money is chosen deliberately: CRR convergence is famously
    # non-monotone ("zig-zag") when the strike lands awkwardly between tree
    # nodes at maturity, so an arbitrary strike can make the error bounce
    # around as N grows even though it trends to zero. ATM avoids that.
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.2
    bs_price = float(bs.price(S, K, T, r, q, sigma, "call"))

    errors = {
        N: abs(bn.price(S, K, T, r, q, sigma, N, "call", american=False) - bs_price)
        for N in [50, 200, 800, 1000]
    }

    assert errors[1000] < 5e-3
    assert errors[50] > errors[200] > errors[800]


def test_american_put_at_least_european_put():
    S, K, T, r, q, sigma, N = 90.0, 100.0, 1.0, 0.05, 0.0, 0.2, 500
    euro = bn.price(S, K, T, r, q, sigma, N, "put", american=False)
    amer = bn.price(S, K, T, r, q, sigma, N, "put", american=True)
    assert amer >= euro


def test_american_put_strictly_exceeds_european_when_early_exercise_valuable():
    # Deep ITM + high rate: holding the put instead of exercising forgoes
    # interest on the strike, which is exactly what makes early exercise
    # valuable for puts (unlike calls).
    S, K, T, r, q, sigma, N = 70.0, 100.0, 1.0, 0.10, 0.0, 0.2, 500
    euro = bn.price(S, K, T, r, q, sigma, N, "put", american=False)
    amer = bn.price(S, K, T, r, q, sigma, N, "put", american=True)
    assert amer > euro + 1.0  # comfortably above any numerical noise


def test_american_call_equals_european_call_when_no_dividends():
    # Classic result: with q = 0, a call's time value is always positive, so
    # continuation value always dominates intrinsic value and early exercise
    # is never optimal. The two trees should produce identical numbers, not
    # just close ones, since the American backward induction's max() should
    # never select the intrinsic branch.
    S, K, T, r, q, sigma, N = 90.0, 100.0, 1.0, 0.07, 0.0, 0.25, 500
    euro = bn.price(S, K, T, r, q, sigma, N, "call", american=False)
    amer = bn.price(S, K, T, r, q, sigma, N, "call", american=True)
    assert amer == pytest.approx(euro, abs=1e-9)


def test_american_call_can_exceed_european_call_with_dividends():
    # With q > 0, holding the call forgoes dividends, so early exercise can
    # become optimal deep ITM -- the mirror image of the put/rate story above.
    S, K, T, r, q, sigma, N = 130.0, 100.0, 1.0, 0.02, 0.08, 0.2, 500
    euro = bn.price(S, K, T, r, q, sigma, N, "call", american=False)
    amer = bn.price(S, K, T, r, q, sigma, N, "call", american=True)
    assert amer > euro


def test_sigma_zero_is_rejected():
    with pytest.raises(ValueError):
        bn.price(100.0, 100.0, 1.0, 0.05, 0.0, 0.0, 100, "call")


def test_expired_tree_is_intrinsic_value():
    assert bn.price(120.0, 100.0, 0.0, 0.05, 0.0, 0.2, 100, "call") == pytest.approx(20.0)
    assert bn.price(80.0, 100.0, 0.0, 0.05, 0.0, 0.2, 100, "put") == pytest.approx(20.0)
