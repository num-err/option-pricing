// Cox-Ross-Rubinstein binomial tree pricing with optional American exercise.
//
// A C++ port of src/pricing/binomial.py -- same recombining lattice, same
// risk-neutral up-probability, same backward induction for American early
// exercise. See the Python module's docstring for the full model writeup
// (why CRR converges to Black-Scholes, and why sigma == 0 is rejected rather
// than silently mishandled). Only scalar inputs are supported, matching the
// Python version.
#pragma once

#include "pricing/types.hpp"

namespace pricing {

// Price a European or American option on an N-step CRR binomial tree.
//
// Throws std::invalid_argument if N < 1, sigma <= 0, or the implied
// risk-neutral probability falls outside (0, 1) for these parameters (the
// tree step would not be arbitrage-free).
double binomial_price(double S, double K, double T, double r, double q, double sigma, int N,
                       OptionType option_type, bool american = false);

}  // namespace pricing
