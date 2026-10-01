# option-pricing

A from-scratch options pricing library: closed-form Black-Scholes-Merton
with analytic Greeks, a Cox-Ross-Rubinstein binomial tree with American
exercise, Monte Carlo simulation with variance reduction, and a Brent's-method
implied volatility solver used to back out a real SPY volatility smile.

Built as an interview portfolio piece, so every function is tested against a
known analytical property (put-call parity, a finite-difference Greek,
convergence to a closed form, a round-trip inversion) rather than a single
hardcoded expected value — the point is to be able to defend every line, not
just to have it pass.

## Quick start

```bash
uv sync
uv run pytest
```

Zero setup beyond `uv` — no API keys, no external services. The one script
that touches the network (`scripts/fetch_chain.py`) already ran once and its
output is committed to `data/spy_chain.csv`, so `scripts/plot_smile.py` and
the figure below reproduce offline.

## What's here

```
src/pricing/
  black_scholes.py    closed-form European price + delta/gamma/vega/theta/rho
  binomial.py          CRR tree, European and American exercise
  monte_carlo.py        naive / antithetic / control-variate GBM simulation
  implied_vol.py        Brent's-method inversion, with no-arbitrage checks
  surface.py            pure chain-filtering + smile-construction helpers
scripts/
  fetch_chain.py         pulls a live SPY chain via yfinance -> data/
  plot_smile.py           builds figures/smile.png from the cached chain
tests/                    one file per pricing module, plus test_parity.py
cpp/                      C++ pricing engine — see below
benchmarks/               numpy vs. C++ throughput benchmarks
```

`src/pricing/*` is intentionally pure — no I/O, no plotting, numpy in and
numpy out — so every pricing function is trivial to unit test and the
network/plotting code in `scripts/` stays separate from the math.

## C++ pricing engine

`src/pricing/*.py` are the reference implementations: validated, readable,
numpy-vectorized. `cpp/` is a from-scratch second implementation in C++,
exposed back to Python as a `pricing_cpp` extension module via
[pybind11](https://github.com/pybind/pybind11). All four pricing modules are
now ported — Monte Carlo, Black-Scholes, the binomial tree, and implied-vol
solving — and each surfaces a different kind of performance or engineering
lesson, documented below rather than asserted.

```
cpp/
  include/pricing/types.hpp         shared OptionType enum
  include/pricing/monte_carlo.hpp   Monte Carlo public API
  include/pricing/black_scholes.hpp Black-Scholes public API
  include/pricing/binomial.hpp      binomial tree public API
  include/pricing/root_finding.hpp  from-scratch Brent's method
  include/pricing/implied_vol.hpp   implied-vol public API
  src/monte_carlo.cpp               single- and multi-threaded estimators
  src/black_scholes.cpp             closed-form price + analytic Greeks
  src/binomial.cpp                  CRR tree, European and American exercise
  src/root_finding.cpp              Brent's method root-finder
  src/implied_vol.cpp               no-arbitrage bounds + Brent inversion
  src/bindings.cpp                  pybind11 module definition
  tests/test_monte_carlo.cpp        standalone correctness check (no Python)
  tests/test_black_scholes.cpp      standalone correctness check (no Python)
  tests/test_binomial.cpp           standalone correctness check (no Python)
  tests/test_root_finding.cpp       standalone correctness check (no Python)
  tests/test_implied_vol.cpp        standalone correctness check (no Python)
```

### Build

```bash
uv sync --group dev                              # installs pybind11
uv run cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
uv run cmake --build cpp/build -j
uv run ctest --test-dir cpp/build --output-on-failure   # correctness (all 5 modules)
uv run python benchmarks/bench_monte_carlo.py            # Monte Carlo throughput
uv run python benchmarks/bench_black_scholes.py          # Black-Scholes throughput
```

### Monte Carlo: what the benchmark found

The first version simply ported the Python loop structure to C++
(`std::mt19937_64` + `std::normal_distribution`, one path per iteration), and
it did **not** beat numpy — it ran at roughly 0.4–0.7x numpy's speed at
every path count tested. Profiling isolated why: `std::normal_distribution`
on libc++ costs about 6x a raw `mt19937_64` draw (~12ns vs ~1.9ns per call,
measured directly), because it draws from the rejection-based Marsaglia
polar method. numpy's `Generator.standard_normal`, by contrast, uses the
Ziggurat algorithm, which only falls back to a transcendental function in
its rare tail case — so numpy's RNG alone is already close to as fast as
this naive C++ loop, before even counting numpy's array-level vectorization.

That ruled out "C++ is faster because it's C++" and pointed at the one
advantage numpy's single-threaded `Generator` genuinely can't match: Monte
Carlo path simulation is embarrassingly parallel. Splitting the same loop
across `std::thread::hardware_concurrency()` threads (8 on the machine these
numbers were measured on) is where the real, reproducible win is:

| estimator | n_paths | numpy | C++ (1 thread) | C++ (8 threads) | speedup vs. numpy |
|---|---:|---:|---:|---:|---:|
| naive | 10,000,000 | 0.192s | 0.288s (0.7x) | 0.060s | 2.6–3.2x |
| antithetic | 10,000,000 | 0.151s | 0.217s (0.7x) | 0.031s | 4.6–6.9x |
| control variate | 10,000,000 | 0.280s | 0.607s (0.5x) | *(not parallelized yet)* | — |

(Run `benchmarks/bench_monte_carlo.py` yourself for current numbers —
timings are single-run wall-clock on one machine, not averaged over
repetitions, and will vary with core count and load. `control_variate`'s
two-pass replay-the-same-stream design wasn't threaded in this pass; that's
the natural next step.)

Both C++ paths are checked against the closed-form Black-Scholes price in
units of their own standard error (the same statistical test
`tests/test_monte_carlo.py` uses), not against numpy's output directly —
two independent random estimators of the same quantity will disagree by a
few standard errors as a matter of course, so bitwise or tight-tolerance
agreement between them would be the wrong thing to assert.

### Black-Scholes: a different kind of result

`cpp/src/black_scholes.cpp` ports `price`/`delta`/`gamma`/`vega`/`theta`/`rho`
plus a `greeks()` that computes all six off one shared d1/d2, including the
same `T == 0` / `sigma == 0` degenerate-case handling as the Python version.
Because the formula is deterministic closed-form (not sampled), the C++ and
Python outputs agree to float64 precision, not just "within a few standard
errors" — `cpp/tests/test_black_scholes.cpp` checks analytic Greeks against
finite differences of price() across a 2,430-point parameter grid (mirroring
`tests/test_black_scholes.py`), both option types, plus the degenerate-input
and deep ITM/OTM limit cases.

Black-Scholes has no loop to parallelize — it's O(1) per option — so the only
way to make the workload large is to price *many* options, which is what a
real vol-surface build or risk run actually does. `benchmarks/bench_black_scholes.py`
prices a chain of random strikes one option at a time through the pybind11
boundary and compares it to numpy pricing the whole chain in one vectorized
call. **numpy wins here, by about 4x at 100,000 options** — the per-option
Python→C++ call overhead dominates the compiled arithmetic underneath it.
This is the inverse of the Monte Carlo lesson: there, numpy paid an
allocation/RNG cost that a tight C++ loop avoided; here, C++ pays a
marshalling cost numpy's single bulk call avoids. The fix is the same shape
as a real integration would use — batch a whole array across the pybind11
boundary in one call instead of one call per option — which is a natural
next step, not yet implemented.

### Binomial tree: a straightforward win, no surprises

`cpp/src/binomial.cpp` ports the CRR lattice and its backward induction
directly — a `std::vector<double>` of node values updated in place,
ascending by index, which is safe here because each node's new value only
depends on the two node values below it from the previous layer, neither of
which has been overwritten yet when it's read (`cpp/include/pricing/binomial.hpp`
spells out the argument). Unlike Monte Carlo, there's no RNG to be the
bottleneck and no array-allocation-per-call for numpy to pay; this is a
plain nested loop against a plain nested loop, and it behaves exactly as
naively expected — `cpp/tests/test_binomial.cpp` reproduces every case from
`tests/test_binomial.py` (European convergence to Black-Scholes, American
puts exceeding European, the q=0 "never exercise a call early" identity,
American calls exceeding European under dividends, the sigma=0 rejection)
and all pass with no numerical surprises worth a benchmark table. Not every
port needs a plot twist; this one is here mainly to complete the engine.

### Implied vol: writing Brent's method instead of calling it

`src/pricing/implied_vol.py` leans on `scipy.optimize.brentq`. The C++ side
has no scipy to call, so `cpp/src/root_finding.cpp` implements Brent's
method from scratch — bisection, the secant method, and inverse quadratic
interpolation, falling back to bisection whenever a trial step would land
outside the bracket or fail to shrink it fast enough. `cpp/tests/test_root_finding.cpp`
validates the root-finder in isolation first (recovering √2, the Dottie
number `cos(x) = x`, and a cubic root away from the bracket midpoint) before
`cpp/src/implied_vol.cpp` relies on it to invert `black_scholes::price` —
same no-arbitrage bounds, same `ArbitrageViolation` exception (registered
with pybind11 via `py::register_exception` so it still subclasses `ValueError`
in Python, matching the Python module's own `ArbitrageViolation(ValueError)`),
same round-trip recovery checked in `cpp/tests/test_implied_vol.cpp` against
the identical parameter grid `tests/test_implied_vol.py` uses. Both
implementations agree to float64 precision, same as Black-Scholes.

`src/pricing/surface.py` (chain-filtering and smile-construction) stays
Python-only — it's I/O-adjacent glue over a pandas DataFrame, not a
numerical kernel, so there's nothing there that would benefit from a port.

## The smile

![SPY implied volatility smile](figures/smile.png)

Two nearest SPY expiries (0.9 and 1.9 calendar days out — SPY lists
near-daily expiries, so "nearest" is always short-dated), spliced from OTM
puts and OTM calls (the liquid side of the chain on each side of the money),
after dropping zero-volume quotes and any quote with a bid-ask spread over
15% of its mid.

## Results

Numbers quoted here come straight from the test suite (`uv run pytest -s`),
not cherry-picked afterward.

**Binomial tree convergence to Black-Scholes** (ATM call, S=K=100, T=1,
r=5%, q=2%, sigma=20%; BS price = 9.227006):

| N steps | tree price | absolute error |
|---:|---:|---:|
| 50 | 9.188225 | 3.88e-2 |
| 200 | 9.217292 | 9.71e-3 |
| 800 | 9.224576 | 2.43e-3 |
| 1000 | 9.225062 | 1.94e-3 |

Error shrinks monotonically and is roughly O(1/N), as expected for CRR. (This
parameter set was chosen deliberately ATM — off-ATM strikes make CRR's
famous non-monotone "zig-zag" convergence show up instead, since the error
also depends on where the strike happens to fall relative to the tree's
terminal nodes at a given N.)

**Monte Carlo variance reduction** (same option, 200,000 paths, one fixed
seed so naive/antithetic/control-variate are directly comparable):

| estimator | standard error | variance reduction vs. naive |
|---|---:|---:|
| naive | 0.030899 | 1.0x |
| antithetic | 0.023007 | 1.80x |
| control variate | 0.012711 | 5.91x |

Control variate wins here because the discounted terminal stock price is
strongly correlated with a near-ATM call payoff (both are increasing
functions of the same terminal draw); antithetic only exploits the weaker
negative correlation between a monotone payoff at `Z` and at `-Z`. All three
estimators landed within 3 standard errors of the closed-form price for both
calls and puts (checked directly via a z-score in `test_monte_carlo.py`).

**The smile itself:** on the near expiry (0.9d), implied vol ranges from
11.7% near the money up to 30.7% for the furthest OTM put; on the 1.9-day
expiry it's 10.5% to 43.4%. A flat 20%-ish surface is what Black-Scholes
would predict for every strike on a given expiry — the market is instead
pricing far-OTM puts several multiples richer than at-the-money.

## Model assumptions and limitations

Black-Scholes-Merton (and the binomial tree, which targets the same
risk-neutral dynamics) assumes:

- **Constant, known volatility** — one sigma for every strike and every
  expiry.
- **Geometric Brownian motion** — returns are lognormal, so the underlying
  never jumps and extreme moves are rarer than they actually are (real
  return distributions have fatter tails).
- **Continuous frictionless hedging** — a market maker can rebalance a
  replicating portfolio continuously at zero transaction cost, which is what
  makes the price genuinely "fair" under the model. Real hedging is
  discrete and costs money, which by itself would already justify some
  compensation baked into the price beyond the textbook formula.
- **Constant r and q** over the life of the option.
- **European exercise unless explicitly relaxed** (the binomial tree's
  `american=True` path is the one place this project actually models early
  exercise).

**Why the smile above is not flat, given those assumptions:** implied
volatility is defined as *whatever sigma makes Black-Scholes match the
market price*. If the market actually priced options under GBM with one
constant sigma, every strike on a given expiry would back out the same
number, and the chart above would be a flat line. It isn't, because:

- **Jump / crash risk.** Equity indices can gap down sharply (1987, 2008,
  2020); GBM's continuous paths assign that near-zero probability. The
  market prices it in anyway, and the cheapest way to buy crash insurance is
  a far-OTM put — so those get bid up far above what a constant-sigma model
  would charge, which is exactly the steep left side of the curve above.
- **Leverage / volatility feedback.** As an equity index falls, the
  remaining equity is thinner relative to the same fixed liabilities, so
  realized volatility tends to rise precisely when the price drops — a
  negative price/vol correlation that constant-sigma GBM has no mechanism
  to represent, but that shows up as richer downside strikes.
- **Supply and demand for tail insurance.** Institutional investors
  structurally buy downside protection (puts) more than they buy or sell
  upside calls, and dealers who write that protection charge a risk premium
  for the hard-to-hedge, discontinuous risk they're taking on — a
  frictional, real-world effect entirely outside a frictionless model.

None of this means Black-Scholes is "wrong" to build or use here — it's the
common language the whole market uses to quote prices as a single number
(the implied vol) instead of a dollar figure, which is exactly why
`implied_vol.py` exists and why the smile above is measured in vol, not
price. It just means that number should be read as *the market's price for
this specific strike and expiry*, not as a forecast of realized volatility,
and the shape of the curve is itself a measurement of exactly which
Black-Scholes assumptions the market is refusing to believe.
