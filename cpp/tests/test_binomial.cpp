// Correctness checks for pricing::binomial_price, mirroring
// tests/test_binomial.py: European convergence to Black-Scholes, American
// vs. European exercise, and the "no early exercise for a call with q=0"
// classic result.
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>

#include "pricing/black_scholes.hpp"
#include "pricing/binomial.hpp"

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
        std::fprintf(stderr, "FAILED %s: actual=%.8f expected=%.8f diff=%.2e\n", label, actual,
                     expected, std::abs(actual - expected));
        ++failures;
    }
}

void check_throws(const std::function<void()>& fn, const char* message) {
    try {
        fn();
        std::fprintf(stderr, "FAILED: %s (expected an exception, none thrown)\n", message);
        ++failures;
    } catch (const std::invalid_argument&) {
        // expected
    }
}

}  // namespace

int main() {
    // European tree converges to Black-Scholes (ATM, to avoid CRR's
    // non-monotone zig-zag convergence off the money).
    {
        double S = 100.0, K = 100.0, T = 1.0, r = 0.05, q = 0.02, sigma = 0.2;
        double bs_price = pricing::price(S, K, T, r, q, sigma, OptionType::Call);

        double e50 = std::abs(pricing::binomial_price(S, K, T, r, q, sigma, 50, OptionType::Call,
                                                        false) -
                               bs_price);
        double e200 = std::abs(
            pricing::binomial_price(S, K, T, r, q, sigma, 200, OptionType::Call, false) -
            bs_price);
        double e800 = std::abs(
            pricing::binomial_price(S, K, T, r, q, sigma, 800, OptionType::Call, false) -
            bs_price);
        double e1000 = std::abs(
            pricing::binomial_price(S, K, T, r, q, sigma, 1000, OptionType::Call, false) -
            bs_price);

        check(e1000 < 5e-3, "N=1000 tree not within 5e-3 of Black-Scholes");
        check(e50 > e200 && e200 > e800, "tree error not monotonically shrinking in N");
    }

    // American put >= European put.
    {
        double S = 90.0, K = 100.0, T = 1.0, r = 0.05, q = 0.0, sigma = 0.2;
        int N = 500;
        double euro = pricing::binomial_price(S, K, T, r, q, sigma, N, OptionType::Put, false);
        double amer = pricing::binomial_price(S, K, T, r, q, sigma, N, OptionType::Put, true);
        check(amer >= euro, "American put priced below European put");
    }

    // Deep ITM put + high rate: early exercise is comfortably valuable.
    {
        double S = 70.0, K = 100.0, T = 1.0, r = 0.10, q = 0.0, sigma = 0.2;
        int N = 500;
        double euro = pricing::binomial_price(S, K, T, r, q, sigma, N, OptionType::Put, false);
        double amer = pricing::binomial_price(S, K, T, r, q, sigma, N, OptionType::Put, true);
        check(amer > euro + 1.0, "American put premium over European too small to be real");
    }

    // q = 0: American call == European call (early exercise never optimal).
    {
        double S = 90.0, K = 100.0, T = 1.0, r = 0.07, q = 0.0, sigma = 0.25;
        int N = 500;
        double euro = pricing::binomial_price(S, K, T, r, q, sigma, N, OptionType::Call, false);
        double amer = pricing::binomial_price(S, K, T, r, q, sigma, N, OptionType::Call, true);
        check_close(amer, euro, 1e-9, "American call should equal European call when q=0");
    }

    // q > 0: American call can exceed European call.
    {
        double S = 130.0, K = 100.0, T = 1.0, r = 0.02, q = 0.08, sigma = 0.2;
        int N = 500;
        double euro = pricing::binomial_price(S, K, T, r, q, sigma, N, OptionType::Call, false);
        double amer = pricing::binomial_price(S, K, T, r, q, sigma, N, OptionType::Call, true);
        check(amer > euro, "American call with dividends should exceed European call");
    }

    // sigma = 0 is rejected.
    check_throws(
        [] { pricing::binomial_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.0, 100, OptionType::Call); },
        "sigma = 0 should throw std::invalid_argument");

    // Expired tree is intrinsic value.
    check_close(pricing::binomial_price(120.0, 100.0, 0.0, 0.05, 0.0, 0.2, 100, OptionType::Call),
                20.0, 1e-9, "expired call is not intrinsic value");
    check_close(pricing::binomial_price(80.0, 100.0, 0.0, 0.05, 0.0, 0.2, 100, OptionType::Put),
                20.0, 1e-9, "expired put is not intrinsic value");

    if (failures > 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
