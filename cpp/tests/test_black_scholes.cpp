// Correctness checks for pricing::price/delta/gamma/vega/theta/rho/greeks,
// mirroring tests/test_black_scholes.py: analytic Greeks must match finite
// differences of price() across a parameter grid, and degenerate inputs
// (T == 0, sigma == 0) must match their documented closed forms exactly.
// Plain asserts + a non-zero exit code on failure, run via ctest.
#include <cmath>
#include <cstdio>
#include <vector>

#include "pricing/black_scholes.hpp"

using pricing::OptionType;

namespace {

int failures = 0;

void check_close(double actual, double expected, double atol, const char* label) {
    if (std::abs(actual - expected) > atol) {
        std::fprintf(stderr, "FAILED %s: actual=%.8f expected=%.8f diff=%.2e (atol=%.1e)\n",
                     label, actual, expected, std::abs(actual - expected), atol);
        ++failures;
    }
}

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        ++failures;
    }
}

struct Params {
    double S, K, T, r, q, sigma;
};

std::vector<Params> grid() {
    std::vector<Params> out;
    for (double S : {60.0, 90.0, 100.0, 110.0, 150.0}) {
        for (double K : {80.0, 100.0, 120.0}) {
            for (double T : {0.25, 1.0, 2.5}) {
                for (double r : {0.0, 0.03, 0.07}) {
                    for (double q : {0.0, 0.02}) {
                        for (double sigma : {0.1, 0.3, 0.6}) {
                            out.push_back({S, K, T, r, q, sigma});
                        }
                    }
                }
            }
        }
    }
    return out;
}

void check_greeks_against_finite_differences(OptionType type) {
    for (const auto& p : grid()) {
        double h_delta = 1e-4;
        double fd_delta =
            (pricing::price(p.S + h_delta, p.K, p.T, p.r, p.q, p.sigma, type) -
             pricing::price(p.S - h_delta, p.K, p.T, p.r, p.q, p.sigma, type)) /
            (2 * h_delta);
        check_close(pricing::delta(p.S, p.K, p.T, p.r, p.q, p.sigma, type), fd_delta, 1e-5,
                    "delta vs finite difference");

        double h_gamma = 1e-2;
        double fd_gamma =
            (pricing::price(p.S + h_gamma, p.K, p.T, p.r, p.q, p.sigma, type) -
             2 * pricing::price(p.S, p.K, p.T, p.r, p.q, p.sigma, type) +
             pricing::price(p.S - h_gamma, p.K, p.T, p.r, p.q, p.sigma, type)) /
            (h_gamma * h_gamma);
        check_close(pricing::gamma(p.S, p.K, p.T, p.r, p.q, p.sigma), fd_gamma, 1e-4,
                    "gamma vs finite difference");

        double h_vega = 1e-5;
        double fd_vega =
            (pricing::price(p.S, p.K, p.T, p.r, p.q, p.sigma + h_vega, type) -
             pricing::price(p.S, p.K, p.T, p.r, p.q, p.sigma - h_vega, type)) /
            (2 * h_vega);
        check_close(pricing::vega(p.S, p.K, p.T, p.r, p.q, p.sigma), fd_vega, 1e-5,
                    "vega vs finite difference");

        double h_theta = 1e-5;
        double fd_theta =
            -(pricing::price(p.S, p.K, p.T + h_theta, p.r, p.q, p.sigma, type) -
              pricing::price(p.S, p.K, p.T - h_theta, p.r, p.q, p.sigma, type)) /
            (2 * h_theta);
        check_close(pricing::theta(p.S, p.K, p.T, p.r, p.q, p.sigma, type), fd_theta, 1e-5,
                    "theta vs finite difference");

        double h_rho = 1e-5;
        double fd_rho =
            (pricing::price(p.S, p.K, p.T, p.r + h_rho, p.q, p.sigma, type) -
             pricing::price(p.S, p.K, p.T, p.r - h_rho, p.q, p.sigma, type)) /
            (2 * h_rho);
        check_close(pricing::rho(p.S, p.K, p.T, p.r, p.q, p.sigma, type), fd_rho, 1e-5,
                    "rho vs finite difference");
    }
}

void check_greeks_struct_matches_individual_calls(OptionType type) {
    for (const auto& p : grid()) {
        auto g = pricing::greeks(p.S, p.K, p.T, p.r, p.q, p.sigma, type);
        check_close(g.price, pricing::price(p.S, p.K, p.T, p.r, p.q, p.sigma, type), 1e-12,
                    "greeks().price vs price()");
        check_close(g.delta, pricing::delta(p.S, p.K, p.T, p.r, p.q, p.sigma, type), 1e-12,
                    "greeks().delta vs delta()");
        check_close(g.gamma, pricing::gamma(p.S, p.K, p.T, p.r, p.q, p.sigma), 1e-12,
                    "greeks().gamma vs gamma()");
        check_close(g.vega, pricing::vega(p.S, p.K, p.T, p.r, p.q, p.sigma), 1e-12,
                    "greeks().vega vs vega()");
        check_close(g.theta, pricing::theta(p.S, p.K, p.T, p.r, p.q, p.sigma, type), 1e-12,
                    "greeks().theta vs theta()");
        check_close(g.rho, pricing::rho(p.S, p.K, p.T, p.r, p.q, p.sigma, type), 1e-12,
                    "greeks().rho vs rho()");
    }
}

void check_degenerate_inputs() {
    // Expired: price is intrinsic value.
    check_close(pricing::price(80.0, 100.0, 0.0, 0.05, 0.02, 0.2, OptionType::Call), 0.0, 1e-12,
                "expired OTM call is worthless");
    check_close(pricing::price(120.0, 100.0, 0.0, 0.05, 0.02, 0.2, OptionType::Call), 20.0, 1e-12,
                "expired ITM call is intrinsic");
    check_close(pricing::price(80.0, 100.0, 0.0, 0.05, 0.02, 0.2, OptionType::Put), 20.0, 1e-12,
                "expired ITM put is intrinsic");

    // Zero vol: deterministic discounted forward payoff.
    double S = 110.0, K = 100.0, T = 1.0, r = 0.05, q = 0.01;
    double forward = S * std::exp((r - q) * T);
    double expected = std::exp(-r * T) * std::max(forward - K, 0.0);
    check_close(pricing::price(S, K, T, r, q, 0.0, OptionType::Call), expected, 1e-10,
                "zero-vol call is discounted deterministic payoff");
    check_close(pricing::delta(S, K, T, r, q, 0.0, OptionType::Call), std::exp(-q * T), 1e-10,
                "zero-vol ITM call delta is exp(-qT)");
    check_close(pricing::delta(90.0, K, T, r, q, 0.0, OptionType::Call), 0.0, 1e-10,
                "zero-vol OTM call delta is 0");

    // Deep ITM / OTM delta limits.
    check_close(pricing::delta(1.0e6, 100.0, 1.0, 0.03, 0.02, 0.2, OptionType::Call),
                std::exp(-0.02 * 1.0), 1e-8, "deep ITM call delta approaches exp(-qT)");
    check_close(pricing::delta(1.0e-3, 100.0, 1.0, 0.03, 0.02, 0.2, OptionType::Call), 0.0, 1e-8,
                "deep OTM call delta approaches 0");
}

void check_price_monotone_in_spot() {
    for (const auto& base : grid()) {
        double prev = pricing::price(50.0, base.K, base.T, base.r, base.q, base.sigma,
                                      OptionType::Call);
        for (double S = 55.0; S <= 200.0; S += 5.0) {
            double cur = pricing::price(S, base.K, base.T, base.r, base.q, base.sigma,
                                         OptionType::Call);
            check(cur > prev, "call price not strictly increasing in spot");
            prev = cur;
        }
    }
}

}  // namespace

int main() {
    check_greeks_against_finite_differences(OptionType::Call);
    check_greeks_against_finite_differences(OptionType::Put);
    check_greeks_struct_matches_individual_calls(OptionType::Call);
    check_greeks_struct_matches_individual_calls(OptionType::Put);
    check_degenerate_inputs();
    check_price_monotone_in_spot();

    if (failures > 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
