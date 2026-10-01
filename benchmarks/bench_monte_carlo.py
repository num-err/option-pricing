"""Benchmark: numpy (src/pricing/monte_carlo.py) vs the C++ pybind11 module
(cpp/src/monte_carlo.cpp) at increasing path counts.

Both implementations run the same estimators for the same option, so the
comparison isolates the cost of the *simulation loop itself* -- numpy's
vectorized array ops vs a C++ loop with O(1) memory -- not a comparison of
numerical methods: both converge to the same Black-Scholes price (see
cpp/tests/test_monte_carlo.cpp and tests/test_monte_carlo.py for the
correctness side of that claim).

Two single-threaded estimators are intentionally included as a negative
result worth keeping: a straightforward port of `naive`/`antithetic` to a
C++ loop using std::mt19937_64 + std::normal_distribution does *not* beat
numpy. Profiling (see README's "What the benchmark found" section) traced
this to std::normal_distribution's trig-heavy Box-Muller implementation
costing ~6x a raw RNG draw, versus numpy's Ziggurat algorithm, which rarely
touches a transcendental function at all. The `_parallel` variants are the
actual payoff: splitting the embarrassingly-parallel path loop across
std::thread::hardware_concurrency() threads is the one axis numpy's
single-threaded Generator can't follow.

Usage
-----
    uv run cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
    uv run cmake --build cpp/build -j
    uv run python benchmarks/bench_monte_carlo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cpp" / "build"))

from pricing import black_scholes as bs  # noqa: E402
from pricing import monte_carlo as mc  # noqa: E402

try:
    import pricing_cpp as pc  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pricing_cpp extension not found -- build it first:\n"
        "  uv run cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release\n"
        "  uv run cmake --build cpp/build -j"
    ) from exc

S, K, T, R, Q, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.02, 0.2
OPTION_TYPE = "call"
PATH_COUNTS = [100_000, 1_000_000, 10_000_000]
SEED = 12345
BS_PRICE = float(bs.price(S, K, T, R, Q, SIGMA, OPTION_TYPE))

ESTIMATORS = [
    ("naive", mc.price_naive, pc.price_naive),
    ("antithetic", mc.price_antithetic, pc.price_antithetic),
    ("control_variate", mc.price_control_variate, pc.price_control_variate),
    ("naive_parallel", mc.price_naive, pc.price_naive_parallel),
    ("antithetic_parallel", mc.price_antithetic, pc.price_antithetic_parallel),
]


def time_call(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return time.perf_counter() - start, result


def main() -> None:
    import os

    print(f"CPU threads available (os.cpu_count): {os.cpu_count()}\n")
    header = f"{'estimator':<20}{'n_paths':>12}{'numpy (s)':>12}{'C++ (s)':>12}{'speedup':>10}"
    print(header)
    print("-" * len(header))

    for name, py_fn, cpp_fn in ESTIMATORS:
        for n_paths in PATH_COUNTS:
            py_time, py_result = time_call(
                py_fn, S, K, T, R, Q, SIGMA, n_paths, OPTION_TYPE, np.random.default_rng(SEED)
            )
            cpp_time, cpp_result = time_call(
                cpp_fn, S, K, T, R, Q, SIGMA, n_paths, pc.OptionType.Call, SEED
            )

            speedup = py_time / cpp_time
            print(
                f"{name:<20}{n_paths:>12,}{py_time:>12.4f}{cpp_time:>12.4f}{speedup:>9.1f}x"
            )

            # Both estimators are independent random draws from the same
            # model, so they won't match bit-for-bit or even to a fixed
            # relative tolerance -- compare each against the known
            # closed-form price in units of *its own* standard error, the
            # same check tests/test_monte_carlo.py and
            # cpp/tests/test_monte_carlo.cpp use for correctness.
            for label, result in (("numpy", py_result), ("C++", cpp_result)):
                z = abs(result.price - BS_PRICE) / result.std_error
                assert z < 5.0, (
                    f"{name} ({label}) at n_paths={n_paths}: price={result.price:.4f} is "
                    f"{z:.1f} standard errors from the Black-Scholes price {BS_PRICE:.4f} -- "
                    "likely a real bug, not sampling noise"
                )
        print()


if __name__ == "__main__":
    main()
