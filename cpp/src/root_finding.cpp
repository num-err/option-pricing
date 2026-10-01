#include "pricing/root_finding.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace pricing {

double brentq(const std::function<double(double)>& f, double a, double b, double xtol,
              double rtol, int max_iter) {
    double fa = f(a);
    double fb = f(b);
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if (fa * fb > 0.0) {
        throw std::invalid_argument("brentq: f(a) and f(b) must have opposite signs");
    }

    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a, fc = fa;
    double d = b;  // unused until the first time mflag is false
    bool mflag = true;

    for (int iter = 0; iter < max_iter; ++iter) {
        const double tol = xtol + rtol * std::abs(b);
        if (fb == 0.0 || std::abs(b - a) < tol) {
            return b;
        }

        double s;
        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation.
            s = a * fb * fc / ((fa - fb) * (fa - fc)) + b * fa * fc / ((fb - fa) * (fb - fc)) +
                c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method.
            s = b - fb * (b - a) / (fb - fa);
        }

        double lo = (3.0 * a + b) / 4.0;
        double hi = b;
        if (lo > hi) std::swap(lo, hi);

        const bool cond1 = s < lo || s > hi;
        const bool cond2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;
        const bool cond3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2.0;
        const bool cond4 = mflag && std::abs(b - c) < tol;
        const bool cond5 = !mflag && std::abs(c - d) < tol;

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = 0.5 * (a + b);  // bisection fallback
            mflag = true;
        } else {
            mflag = false;
        }

        const double fs = f(s);
        d = c;
        c = b;
        fc = fb;
        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }

    throw std::runtime_error("brentq: max_iter exceeded without converging");
}

}  // namespace pricing
