#include "pricing/binomial.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pricing {

namespace {

inline double intrinsic(double S, double K, OptionType type) {
    return type == OptionType::Call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
}

}  // namespace

double binomial_price(double S, double K, double T, double r, double q, double sigma, int N,
                       OptionType option_type, bool american) {
    if (N < 1) {
        throw std::invalid_argument("N must be >= 1");
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument("binomial_price requires sigma > 0 (see header comment)");
    }
    if (T <= 0.0) {
        return intrinsic(S, K, option_type);
    }

    const double dt = T / N;
    const double x = sigma * std::sqrt(dt);  // log up-move size: u = exp(x), d = exp(-x)
    const double disc = std::exp(-r * dt);
    const double up = std::exp(x);
    const double down = std::exp(-x);
    const double p = (std::exp((r - q) * dt) - down) / (up - down);
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument(
            "risk-neutral probability outside (0, 1) for these parameters; the tree step is "
            "not arbitrage-free (try a larger N)");
    }

    // Terminal spot prices: j up-moves and (N - j) down-moves, j = 0..N.
    // Written as S * exp(x * (2j - N)) rather than S * u^j * d^(N-j) to
    // avoid accumulating floating-point error from repeated powers of u, d.
    std::vector<double> values(N + 1);
    for (int j = 0; j <= N; ++j) {
        double S_terminal = S * std::exp(x * (2 * j - N));
        values[j] = intrinsic(S_terminal, K, option_type);
    }

    for (int step = N - 1; step >= 0; --step) {
        // Ascending-index in-place update is safe: values[j]'s new value
        // only depends on values[j] and values[j+1] from the layer below,
        // neither of which has been overwritten yet when we compute it.
        for (int j = 0; j <= step; ++j) {
            values[j] = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
        }
        if (american) {
            for (int j = 0; j <= step; ++j) {
                double S_step = S * std::exp(x * (2 * j - step));
                values[j] = std::max(values[j], intrinsic(S_step, K, option_type));
            }
        }
    }

    return values[0];
}

}  // namespace pricing
