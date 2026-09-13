"""Put-call parity: C - P = S*exp(-qT) - K*exp(-rT).

This holds by a static replication argument alone (long call + short put has
the same payoff as a forward), independent of the distributional assumptions
behind Black-Scholes. It is the sharpest possible check that the call and put
formulas are internally consistent with each other.
"""

import itertools

import numpy as np

from pricing import black_scholes as bs

S_GRID = [50.0, 80.0, 100.0, 120.0, 200.0]
K_GRID = [60.0, 100.0, 150.0]
T_GRID = [0.1, 0.5, 1.0, 3.0]
R_GRID = [0.0, 0.02, 0.08]
Q_GRID = [0.0, 0.01, 0.05]
SIGMA_GRID = [0.05, 0.2, 0.5, 1.0]


def test_put_call_parity_holds_on_a_grid():
    combos = list(itertools.product(S_GRID, K_GRID, T_GRID, R_GRID, Q_GRID, SIGMA_GRID))
    S, K, T, r, q, sigma = (np.array(x) for x in zip(*combos))

    call = bs.price(S, K, T, r, q, sigma, option_type="call")
    put = bs.price(S, K, T, r, q, sigma, option_type="put")

    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

    np.testing.assert_allclose(lhs, rhs, atol=1e-9, rtol=1e-9)
