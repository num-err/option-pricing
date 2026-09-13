"""Tests for pricing.monte_carlo: statistical correctness (within a few
standard errors of the closed-form price) and effectiveness of variance
reduction (materially smaller standard error at the same path count).
"""

import numpy as np
import pytest

from pricing import black_scholes as bs
from pricing import monte_carlo as mc

S, K, T, R, Q, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.02, 0.2
N_PATHS = 200_000

ESTIMATORS = {
    "naive": mc.price_naive,
    "antithetic": mc.price_antithetic,
    "control_variate": mc.price_control_variate,
}


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("name", ESTIMATORS)
def test_estimator_within_three_standard_errors_of_black_scholes(name, option_type):
    fn = ESTIMATORS[name]
    bs_price = float(bs.price(S, K, T, R, Q, SIGMA, option_type))
    rng = np.random.default_rng(12345)
    result = fn(S, K, T, R, Q, SIGMA, N_PATHS, option_type, rng)

    z = abs(result.price - bs_price) / result.std_error
    assert z < 3.0, (
        f"{name} {option_type}: price={result.price:.5f} vs BS={bs_price:.5f}, "
        f"z={z:.2f} standard errors"
    )


def test_variance_reduction_beats_naive():
    option_type = "call"
    seed = 2024

    naive = mc.price_naive(S, K, T, R, Q, SIGMA, N_PATHS, option_type, np.random.default_rng(seed))
    antithetic = mc.price_antithetic(
        S, K, T, R, Q, SIGMA, N_PATHS, option_type, np.random.default_rng(seed)
    )
    control = mc.price_control_variate(
        S, K, T, R, Q, SIGMA, N_PATHS, option_type, np.random.default_rng(seed)
    )

    antithetic_factor = (naive.std_error / antithetic.std_error) ** 2
    control_factor = (naive.std_error / control.std_error) ** 2

    print(f"\nVariance reduction at n_paths={N_PATHS}, {option_type} option:")
    print(f"  naive           se={naive.std_error:.6f}")
    print(f"  antithetic      se={antithetic.std_error:.6f}  factor={antithetic_factor:.2f}x")
    print(f"  control variate se={control.std_error:.6f}  factor={control_factor:.2f}x")

    # "Materially below": require at least a 25% variance reduction, well
    # under what either method actually achieves here, to leave headroom for
    # sampling noise in the factor itself while still being a real test.
    assert antithetic.std_error < naive.std_error
    assert antithetic_factor > 1.25
    assert control.std_error < naive.std_error
    assert control_factor > 1.25


def test_reproducible_with_same_seed():
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    r1 = mc.price_naive(S, K, T, R, Q, SIGMA, 10_000, "call", rng1)
    r2 = mc.price_naive(S, K, T, R, Q, SIGMA, 10_000, "call", rng2)
    assert r1.price == r2.price
    assert r1.std_error == r2.std_error
