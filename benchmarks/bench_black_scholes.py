"""Benchmark: numpy (src/pricing/black_scholes.py) vs the C++ pybind11 module
(cpp/src/black_scholes.cpp) pricing a whole chain of options.

Unlike Monte Carlo, Black-Scholes is O(1) per option -- there's no iterative
simulation to parallelize, no RNG, no variance to reduce. The only way to
make this workload large is to price many options at once, which is exactly
what a real vol-surface build or a risk run does: thousands of strikes x
expiries, each needing price + all five Greeks. That reframes the
comparison from "iterate faster" (Monte Carlo's story) to "call overhead and
branching, loop vs. broadcast" -- numpy prices the whole chain in one
vectorized call with no per-option Python overhead, while the C++ side pays
a real Python-to-C++ call per option unless batched. Both are measured here
to make that distinction concrete rather than asserted.

Usage
-----
    uv run cmake --build cpp/build -j   # build pricing_cpp if not already built
    uv run python benchmarks/bench_black_scholes.py
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

try:
    import pricing_cpp as pc  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pricing_cpp extension not found -- build it first:\n"
        "  uv run cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release\n"
        "  uv run cmake --build cpp/build -j"
    ) from exc

N_OPTIONS_LIST = [1_000, 10_000, 100_000]
R, Q = 0.05, 0.02


def make_chain(n_options: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    S = rng.uniform(80.0, 120.0, n_options)
    K = rng.uniform(80.0, 120.0, n_options)
    T = rng.uniform(0.1, 2.0, n_options)
    sigma = rng.uniform(0.1, 0.6, n_options)
    return S, K, T, sigma


def main() -> None:
    header = f"{'n_options':>12}{'numpy price':>14}{'numpy greeks':>14}{'C++ price':>12}{'C++ greeks':>12}"
    print(header)
    print("-" * len(header))

    for n_options in N_OPTIONS_LIST:
        S, K, T, sigma = make_chain(n_options)

        t0 = time.perf_counter()
        py_price = bs.price(S, K, T, R, Q, sigma, "call")
        t_py_price = time.perf_counter() - t0

        t0 = time.perf_counter()
        bs.delta(S, K, T, R, Q, sigma, "call")
        bs.gamma(S, K, T, R, Q, sigma)
        bs.vega(S, K, T, R, Q, sigma)
        bs.theta(S, K, T, R, Q, sigma, "call")
        bs.rho(S, K, T, R, Q, sigma, "call")
        t_py_greeks = time.perf_counter() - t0

        t0 = time.perf_counter()
        cpp_price = np.empty(n_options)
        for i in range(n_options):
            cpp_price[i] = pc.price(S[i], K[i], T[i], R, Q, sigma[i], pc.OptionType.Call)
        t_cpp_price = time.perf_counter() - t0

        t0 = time.perf_counter()
        for i in range(n_options):
            pc.greeks(S[i], K[i], T[i], R, Q, sigma[i], pc.OptionType.Call)
        t_cpp_greeks = time.perf_counter() - t0

        print(
            f"{n_options:>12,}{t_py_price:>14.5f}{t_py_greeks:>14.5f}"
            f"{t_cpp_price:>12.5f}{t_cpp_greeks:>12.5f}"
        )

        max_diff = np.max(np.abs(cpp_price - py_price))
        assert max_diff < 1e-9, f"numpy and C++ prices disagree by {max_diff:.2e} -- real bug"

    print(
        "\nBoth implementations compute identical closed-form formulas, so prices match to "
        "~1e-12\n(see the assert above) -- the comparison here is pure call/loop overhead, not "
        "numerical\nmethod. The per-option Python<->C++ call in this loop is the dominant cost "
        "on the C++\nside; a real integration would batch a whole array across the pybind11 "
        "boundary in\none call instead, which this benchmark deliberately does not do."
    )


if __name__ == "__main__":
    main()
