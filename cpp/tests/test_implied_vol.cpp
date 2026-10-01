// Correctness checks for pricing::implied_vol, mirroring
// tests/test_implied_vol.py: round-trip recovery of a known volatility, and
// that arbitrage-violating prices throw rather than returning garbage.
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "pricing/black_scholes.hpp"
#include "pricing/implied_vol.hpp"

using pricing::OptionType;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        ++failures;
    }
}

void check_close(double actual, double expected, double atol, const char* label) {
    if (std::abs(actual - expected) > atol) {
        std::fprintf(stderr, "FAILED %s: actual=%.10f expected=%.10f diff=%.2e\n", label, actual,
                     expected, std::abs(actual - expected));
        ++failures;
    }
}

void check_throws_arbitrage(const std::function<void()>& fn, const char* message) {
    try {
        fn();
        std::fprintf(stderr, "FAILED: %s (expected ArbitrageViolation, none thrown)\n", message);
        ++failures;
    } catch (const pricing::ArbitrageViolation&) {
        // expected
    }
}

// Below this vega, recovering sigma to 1e-6 is not numerically meaningful
// (price is essentially flat in sigma there) -- see
// tests/test_implied_vol.py's MIN_VEGA_FOR_STABLE_INVERSION for the full
// rationale. Same threshold used here for the same reason.
constexpr double MIN_VEGA_FOR_STABLE_INVERSION = 1e-2;

}  // namespace

int main() {
    // Round-trip: price a known sigma, recover it via implied_vol.
    {
        std::vector<double> S_grid = {80.0, 100.0, 120.0};
        std::vector<double> K_grid = {80.0, 100.0, 120.0};
        std::vector<double> T_grid = {0.1, 1.0, 2.0};
        std::vector<double> r_grid = {0.0, 0.03, 0.08};
        std::vector<double> q_grid = {0.0, 0.02};
        std::vector<double> sigma_grid = {0.05, 0.2, 0.5, 1.2};

        for (OptionType type : {OptionType::Call, OptionType::Put}) {
            int checked = 0;
            for (double S : S_grid) {
                for (double K : K_grid) {
                    for (double T : T_grid) {
                        for (double r : r_grid) {
                            for (double q : q_grid) {
                                for (double sigma_true : sigma_grid) {
                                    if (pricing::vega(S, K, T, r, q, sigma_true) <
                                        MIN_VEGA_FOR_STABLE_INVERSION) {
                                        continue;
                                    }
                                    double p = pricing::price(S, K, T, r, q, sigma_true, type);
                                    double recovered =
                                        pricing::implied_vol(p, S, K, T, r, q, type);
                                    check_close(recovered, sigma_true, 1e-6,
                                                "implied_vol round-trip");
                                    ++checked;
                                }
                            }
                        }
                    }
                }
            }
            check(checked > 400, "vega filter left too few cases checked (sanity check)");
        }
    }

    // Price above the upper no-arbitrage bound raises.
    {
        double S = 100.0, K = 100.0, T = 1.0, r = 0.05, q = 0.0;
        auto bounds = pricing::implied_vol_price_bounds(S, K, T, r, q, OptionType::Call);
        check_throws_arbitrage(
            [&] { pricing::implied_vol(bounds.upper + 1.0, S, K, T, r, q, OptionType::Call); },
            "price above upper bound should raise ArbitrageViolation");
    }

    // Price below the lower no-arbitrage bound raises.
    {
        double S = 100.0, K = 120.0, T = 1.0, r = 0.05, q = 0.0;
        auto bounds = pricing::implied_vol_price_bounds(S, K, T, r, q, OptionType::Call);
        check_close(bounds.lower, 0.0, 1e-9, "lower bound should be 0 for this far-OTM call");
        check_throws_arbitrage(
            [&] { pricing::implied_vol(-1.0, S, K, T, r, q, OptionType::Call); },
            "price below lower bound should raise ArbitrageViolation");
    }

    // Negative put price raises.
    check_throws_arbitrage(
        [] { pricing::implied_vol(-0.5, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Put); },
        "negative put price should raise ArbitrageViolation");

    // T <= 0 raises std::invalid_argument (ArbitrageViolation derives from
    // it, so this also confirms the catch-as-base-class relationship holds).
    {
        bool threw = false;
        try {
            pricing::implied_vol(5.0, 100.0, 100.0, 0.0, 0.05, 0.0, OptionType::Call);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "T <= 0 should raise std::invalid_argument");
    }

    if (failures > 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
