// Implied volatility recovery via Brent's method.
//
// A C++ port of src/pricing/implied_vol.py -- same no-arbitrage bounds, same
// ArbitrageViolation-on-bad-quote behavior, same root-find of
// black_scholes::price over sigma. See the Python module's docstring for the
// full argument for why price is monotone in sigma and why out-of-bounds
// quotes are rejected rather than silently extrapolated. The root-find here
// uses pricing::brentq (root_finding.hpp), a from-scratch Brent's method
// implementation, rather than a binding to scipy.
#pragma once

#include <stdexcept>
#include <string>

#include "pricing/types.hpp"

namespace pricing {

// A quoted price falls outside the model-free no-arbitrage bounds, so no
// volatility (however large or small) could have produced it. Derives from
// std::invalid_argument so callers that only expect that can still catch
// it, mirroring the Python ArbitrageViolation(ValueError) relationship.
class ArbitrageViolation : public std::invalid_argument {
public:
    explicit ArbitrageViolation(const std::string& message) : std::invalid_argument(message) {}
};

struct PriceBounds {
    double lower;
    double upper;
};

// Model-free no-arbitrage bounds for a European price.
PriceBounds implied_vol_price_bounds(double S, double K, double T, double r, double q,
                                      OptionType option_type);

// Inverts black_scholes::price for the volatility consistent with
// `target_price`. T must be > 0. Throws ArbitrageViolation if target_price
// is outside the no-arbitrage bounds, or outside the price range spanned by
// [sigma_low, sigma_high]; throws std::invalid_argument if T <= 0.
double implied_vol(double target_price, double S, double K, double T, double r, double q,
                    OptionType option_type, double sigma_low = 1e-6, double sigma_high = 5.0);

}  // namespace pricing
