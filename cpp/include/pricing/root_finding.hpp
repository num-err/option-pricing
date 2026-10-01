// Brent's method for scalar root-finding: combines bisection, the secant
// method, and inverse quadratic interpolation, falling back to bisection
// whenever the faster methods would step outside the current bracket or
// fail to make adequate progress. Converges superlinearly on smooth
// functions without needing a derivative -- this is what scipy.optimize.brentq
// (used by src/pricing/implied_vol.py) wraps; this is a from-scratch
// implementation for the C++ side rather than a binding to it.
#pragma once

#include <functional>

namespace pricing {

// Finds a root of f in [a, b], where f(a) and f(b) must have opposite signs
// (or one of them must already be zero). Iterates until the bracket width
// is below `xtol + rtol * |b|` or `max_iter` iterations are exhausted.
//
// Throws std::invalid_argument if f(a) and f(b) have the same sign, and
// std::runtime_error if max_iter is exceeded without converging.
double brentq(const std::function<double(double)>& f, double a, double b, double xtol = 1e-12,
              double rtol = 1e-12, int max_iter = 200);

}  // namespace pricing
