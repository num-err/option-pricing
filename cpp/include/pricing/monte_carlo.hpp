// Monte Carlo pricing of European payoffs under GBM, with variance reduction.
//
// This is a performance-oriented C++ port of src/pricing/monte_carlo.py --
// same model, same three estimators, same closed-form terminal-price
// sampling (no path discretization, since a European payoff only depends on
// S_T). See the Python module's docstring for the full model writeup; this
// header only documents what differs in the port.
//
// Numerics: each estimator streams one double at a time through Welford's
// online algorithm for mean and variance, so memory use is O(1) regardless
// of n_paths -- the Python/numpy version instead materializes an n_paths
// array per call. That's the main source of any speedup: not that the
// arithmetic itself is faster, but that there's no array allocation,
// bounds-checked indexing, or temporary-array churn in the hot loop.
#pragma once

#include <cstdint>

#include "pricing/types.hpp"

namespace pricing {

struct MCResult {
    double price;
    double std_error;
};

// Naive Monte Carlo: n_paths independent terminal draws.
MCResult price_naive(double S, double K, double T, double r, double q, double sigma,
                      std::int64_t n_paths, OptionType option_type, std::uint64_t seed);

// Antithetic variates: n_paths total simulated draws (n_paths / 2 pairs),
// so the compute budget matches price_naive(n_paths=...) exactly.
MCResult price_antithetic(double S, double K, double T, double r, double q, double sigma,
                           std::int64_t n_paths, OptionType option_type, std::uint64_t seed);

// Control variate: the discounted terminal stock price exp(-rT) S_T, whose
// expectation S*exp(-qT) is known in closed form. Two passes over n_paths
// draws (first to estimate the optimal coefficient b, second to apply it),
// replaying the identical random sequence rather than storing it, to stay
// O(1) in memory.
MCResult price_control_variate(double S, double K, double T, double r, double q, double sigma,
                                std::int64_t n_paths, OptionType option_type, std::uint64_t seed);

// Thread-parallel naive Monte Carlo: splits n_paths evenly across
// std::thread::hardware_concurrency() threads, each with its own RNG stream
// seeded by `seed` offset by a large odd constant per thread index (not a
// cryptographically rigorous stream split, but sufficient decorrelation for
// variance-reduction purposes in a single pricing run). Per-thread Welford
// moments are combined with Chan's parallel-variance formula, so the result
// is the same statistic price_naive would compute, not an approximation.
MCResult price_naive_parallel(double S, double K, double T, double r, double q, double sigma,
                               std::int64_t n_paths, OptionType option_type, std::uint64_t seed);

// Thread-parallel antithetic variates, same threading strategy as
// price_naive_parallel applied to antithetic pairs instead of single draws.
MCResult price_antithetic_parallel(double S, double K, double T, double r, double q, double sigma,
                                    std::int64_t n_paths, OptionType option_type,
                                    std::uint64_t seed);

}  // namespace pricing
