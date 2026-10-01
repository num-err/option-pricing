// Correctness checks for pricing::price_naive/antithetic/control_variate,
// mirroring tests/test_monte_carlo.py: each estimator must land within a few
// standard errors of the closed-form Black-Scholes price, and the
// variance-reduction estimators must materially beat naive at the same path
// count. Plain asserts + a non-zero exit code on failure, run via ctest --
// no GoogleTest dependency, to keep the build self-contained.
//
// The Black-Scholes reference below is a standalone copy (erf-based normal
// CDF) for this test only, not a port of src/pricing/black_scholes.py --
// that port is a separate, not-yet-done step.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <utility>

#include "pricing/monte_carlo.hpp"

using pricing::OptionType;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        ++failures;
    }
}

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double black_scholes_price(double S, double K, double T, double r, double q, double sigma,
                            OptionType type) {
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    if (type == OptionType::Call) {
        return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    }
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
}

}  // namespace

int main() {
    const double S = 100.0, K = 100.0, T = 1.0, R = 0.05, Q = 0.02, SIGMA = 0.2;
    const std::int64_t N_PATHS = 200000;
    const std::uint64_t SEED = 12345;

    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        double bs_price = black_scholes_price(S, K, T, R, Q, SIGMA, type);

        auto naive = pricing::price_naive(S, K, T, R, Q, SIGMA, N_PATHS, type, SEED);
        auto antithetic = pricing::price_antithetic(S, K, T, R, Q, SIGMA, N_PATHS, type, SEED);
        auto control = pricing::price_control_variate(S, K, T, R, Q, SIGMA, N_PATHS, type, SEED);
        auto naive_par = pricing::price_naive_parallel(S, K, T, R, Q, SIGMA, N_PATHS, type, SEED);
        auto antithetic_par =
            pricing::price_antithetic_parallel(S, K, T, R, Q, SIGMA, N_PATHS, type, SEED);

        for (auto [name, result] :
             {std::pair{"naive", naive}, std::pair{"antithetic", antithetic},
              std::pair{"control_variate", control}, std::pair{"naive_parallel", naive_par},
              std::pair{"antithetic_parallel", antithetic_par}}) {
            double z = std::abs(result.price - bs_price) / result.std_error;
            std::printf("%-16s type=%-4s price=%.6f bs=%.6f se=%.6f z=%.2f\n",
                        type == OptionType::Call ? "call" : "put", name, result.price, bs_price,
                        result.std_error, z);
            check(z < 4.0, "estimator price not within 4 standard errors of Black-Scholes");
        }

        double antithetic_factor = (naive.std_error / antithetic.std_error);
        double control_factor = (naive.std_error / control.std_error);
        antithetic_factor *= antithetic_factor;
        control_factor *= control_factor;
        check(antithetic.std_error < naive.std_error, "antithetic did not reduce variance");
        check(control.std_error < naive.std_error, "control variate did not reduce variance");
        check(antithetic_factor > 1.25, "antithetic variance reduction factor too small");
        check(control_factor > 1.25, "control variate variance reduction factor too small");
    }

    if (failures > 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
