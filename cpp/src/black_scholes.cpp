#include "pricing/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace pricing {

namespace {

inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

inline double norm_pdf(double x) {
    static const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

inline double intrinsic(double S, double K, OptionType type) {
    return type == OptionType::Call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
}

// Only meaningful where T > 0 and sigma > 0; callers must guard against the
// degenerate cases (handled explicitly in every public function below)
// before calling this.
std::pair<double, double> d1_d2(double S, double K, double T, double r, double q, double sigma) {
    double vol_sqrt_t = sigma * std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t;
    double d2 = d1 - vol_sqrt_t;
    return {d1, d2};
}

}  // namespace

double price(double S, double K, double T, double r, double q, double sigma,
             OptionType option_type) {
    if (T <= 0.0) return intrinsic(S, K, option_type);
    if (sigma <= 0.0) {
        double forward = S * std::exp((r - q) * T);
        return std::exp(-r * T) * intrinsic(forward, K, option_type);
    }
    auto [d1, d2] = d1_d2(S, K, T, r, q, sigma);
    if (option_type == OptionType::Call) {
        return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    }
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
}

double delta(double S, double K, double T, double r, double q, double sigma,
             OptionType option_type) {
    if (T <= 0.0) {
        if (option_type == OptionType::Call) return S > K ? 1.0 : 0.0;
        return S < K ? -1.0 : 0.0;
    }
    if (sigma <= 0.0) {
        double forward = S * std::exp((r - q) * T);
        if (option_type == OptionType::Call) return forward > K ? std::exp(-q * T) : 0.0;
        return forward < K ? -std::exp(-q * T) : 0.0;
    }
    auto [d1, d2] = d1_d2(S, K, T, r, q, sigma);
    (void)d2;
    if (option_type == OptionType::Call) return std::exp(-q * T) * norm_cdf(d1);
    return -std::exp(-q * T) * norm_cdf(-d1);
}

double gamma(double S, double K, double T, double r, double q, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    auto [d1, d2] = d1_d2(S, K, T, r, q, sigma);
    (void)d2;
    return std::exp(-q * T) * norm_pdf(d1) / (S * sigma * std::sqrt(T));
}

double vega(double S, double K, double T, double r, double q, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    auto [d1, d2] = d1_d2(S, K, T, r, q, sigma);
    (void)d2;
    return S * std::exp(-q * T) * norm_pdf(d1) * std::sqrt(T);
}

double theta(double S, double K, double T, double r, double q, double sigma,
             OptionType option_type) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    auto [d1, d2] = d1_d2(S, K, T, r, q, sigma);
    double decay_term = -S * std::exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * std::sqrt(T));
    if (option_type == OptionType::Call) {
        return decay_term + q * S * std::exp(-q * T) * norm_cdf(d1) -
               r * K * std::exp(-r * T) * norm_cdf(d2);
    }
    return decay_term - q * S * std::exp(-q * T) * norm_cdf(-d1) +
           r * K * std::exp(-r * T) * norm_cdf(-d2);
}

double rho(double S, double K, double T, double r, double q, double sigma,
           OptionType option_type) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    auto [d1, d2] = d1_d2(S, K, T, r, q, sigma);
    (void)d1;
    if (option_type == OptionType::Call) return K * T * std::exp(-r * T) * norm_cdf(d2);
    return -K * T * std::exp(-r * T) * norm_cdf(-d2);
}

Greeks greeks(double S, double K, double T, double r, double q, double sigma,
              OptionType option_type) {
    Greeks g{};

    if (T <= 0.0) {
        g.price = intrinsic(S, K, option_type);
        g.delta = option_type == OptionType::Call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0);
        return g;
    }
    if (sigma <= 0.0) {
        double forward = S * std::exp((r - q) * T);
        g.price = std::exp(-r * T) * intrinsic(forward, K, option_type);
        if (option_type == OptionType::Call) {
            g.delta = forward > K ? std::exp(-q * T) : 0.0;
        } else {
            g.delta = forward < K ? -std::exp(-q * T) : 0.0;
        }
        return g;
    }

    auto [d1, d2] = d1_d2(S, K, T, r, q, sigma);
    const double disc_q = std::exp(-q * T);
    const double disc_r = std::exp(-r * T);
    const double pdf_d1 = norm_pdf(d1);
    const double sqrt_t = std::sqrt(T);

    if (option_type == OptionType::Call) {
        g.price = S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2);
        g.delta = disc_q * norm_cdf(d1);
        g.theta = -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t) + q * S * disc_q * norm_cdf(d1) -
                  r * K * disc_r * norm_cdf(d2);
        g.rho = K * T * disc_r * norm_cdf(d2);
    } else {
        g.price = K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1);
        g.delta = -disc_q * norm_cdf(-d1);
        g.theta = -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t) -
                  q * S * disc_q * norm_cdf(-d1) + r * K * disc_r * norm_cdf(-d2);
        g.rho = -K * T * disc_r * norm_cdf(-d2);
    }
    g.gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t);
    g.vega = S * disc_q * pdf_d1 * sqrt_t;
    return g;
}

}  // namespace pricing
