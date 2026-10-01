// Correctness checks for pricing::brentq against known roots, independent of
// any finance use case -- this validates the root-finder itself before
// implied_vol.cpp relies on it.
#include <cmath>
#include <cstdio>

#include "pricing/root_finding.hpp"

namespace {

int failures = 0;

void check_close(double actual, double expected, double atol, const char* label) {
    if (std::abs(actual - expected) > atol) {
        std::fprintf(stderr, "FAILED %s: actual=%.12f expected=%.12f diff=%.2e\n", label, actual,
                     expected, std::abs(actual - expected));
        ++failures;
    }
}

}  // namespace

int main() {
    // x^2 - 2 = 0 on [0, 2] -> sqrt(2).
    double root1 = pricing::brentq([](double x) { return x * x - 2.0; }, 0.0, 2.0);
    check_close(root1, std::sqrt(2.0), 1e-10, "sqrt(2) via x^2 - 2");

    // cos(x) - x = 0 on [0, 1] -> the Dottie number, a classic Brent's-method
    // test case (transcendental, not symmetric around the root).
    double root2 = pricing::brentq([](double x) { return std::cos(x) - x; }, 0.0, 1.0);
    check_close(root2, 0.7390851332151607, 1e-10, "Dottie number via cos(x) - x");

    // A cubic with a root far from the bracket midpoint, to exercise the
    // inverse-quadratic/secant branches rather than pure bisection.
    double root3 =
        pricing::brentq([](double x) { return (x - 1.0) * (x - 1.0) * (x - 1.0) - 1e-6; }, 0.5,
                         5.0, 1e-13, 1e-13);
    check_close(root3, 1.0 + std::cbrt(1e-6), 1e-9, "cubic root near x=1");

    // A root already at an endpoint should return immediately.
    double root4 = pricing::brentq([](double x) { return x - 3.0; }, 3.0, 10.0);
    check_close(root4, 3.0, 1e-12, "root exactly at lower endpoint");

    if (failures > 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
