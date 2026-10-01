// Closed-form Black-Scholes-Merton prices and analytic Greeks.
//
// A C++ port of src/pricing/black_scholes.py -- same model, same formulas,
// same degenerate-input handling (T == 0 and sigma == 0, documented in the
// Python module's docstring). The Python version is vectorized over numpy
// arrays; this port is scalar (one option at a time), since C++'s payoff
// here is per-call overhead and compiled arithmetic rather than array
// vectorization -- see benchmarks/bench_black_scholes.py for pricing many
// options in a loop vs. numpy's broadcasting.
#pragma once

#include "pricing/types.hpp"

namespace pricing {

struct Greeks {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

// European option price under Black-Scholes-Merton.
double price(double S, double K, double T, double r, double q, double sigma,
             OptionType option_type);

// Sensitivity of price to the spot price, dV/dS.
double delta(double S, double K, double T, double r, double q, double sigma,
             OptionType option_type);

// Curvature of price with respect to spot, d^2V/dS^2. Identical for calls and puts.
double gamma(double S, double K, double T, double r, double q, double sigma);

// Sensitivity of price to volatility, dV/dsigma. Identical for calls and puts.
double vega(double S, double K, double T, double r, double q, double sigma);

// Sensitivity of price to the passage of calendar time, dV/dt = -dV/dT.
double theta(double S, double K, double T, double r, double q, double sigma,
             OptionType option_type);

// Sensitivity of price to the risk-free rate, dV/dr.
double rho(double S, double K, double T, double r, double q, double sigma,
           OptionType option_type);

// All of the above in one call, sharing a single d1/d2 computation --
// cheaper than calling price/delta/gamma/vega/theta/rho separately when all
// six are wanted for the same option.
Greeks greeks(double S, double K, double T, double r, double q, double sigma,
              OptionType option_type);

}  // namespace pricing
