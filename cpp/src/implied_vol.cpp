#include "pricing/implied_vol.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "pricing/black_scholes.hpp"
#include "pricing/root_finding.hpp"

namespace pricing {

PriceBounds implied_vol_price_bounds(double S, double K, double T, double r, double q,
                                      OptionType option_type) {
    const double disc_S = S * std::exp(-q * T);
    const double disc_K = K * std::exp(-r * T);
    if (option_type == OptionType::Call) {
        return {std::max(disc_S - disc_K, 0.0), disc_S};
    }
    return {std::max(disc_K - disc_S, 0.0), disc_K};
}

double implied_vol(double target_price, double S, double K, double T, double r, double q,
                    OptionType option_type, double sigma_low, double sigma_high) {
    if (T <= 0.0) {
        throw std::invalid_argument("implied_vol is undefined for T <= 0 (no time value to invert)");
    }

    const PriceBounds bounds = implied_vol_price_bounds(S, K, T, r, q, option_type);
    const double tol = 1e-10 * std::max(1.0, bounds.upper);
    if (target_price < bounds.lower - tol || target_price > bounds.upper + tol) {
        std::ostringstream msg;
        msg << "price=" << target_price << " violates no-arbitrage bounds [" << bounds.lower
            << ", " << bounds.upper << "]";
        throw ArbitrageViolation(msg.str());
    }

    auto objective = [&](double sigma) {
        return pricing::price(S, K, T, r, q, sigma, option_type) - target_price;
    };

    const double f_lo = objective(sigma_low);
    const double f_hi = objective(sigma_high);
    if (f_lo > 0.0 || f_hi < 0.0) {
        throw ArbitrageViolation(
            "price is within the theoretical no-arbitrage bounds but outside the price range "
            "spanned by [sigma_low, sigma_high]; pass a wider bracket");
    }

    return brentq(objective, sigma_low, sigma_high, 1e-12, 1e-12);
}

}  // namespace pricing
