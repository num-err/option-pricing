#include "pricing/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace pricing {

namespace {

inline double intrinsic(double S, double K, OptionType type) {
    return type == OptionType::Call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
}

void validate(std::int64_t n_paths) {
    if (n_paths < 2) {
        throw std::invalid_argument("n_paths must be >= 2 to estimate a standard error");
    }
}

// Welford's online algorithm: mean and variance in one pass, O(1) memory.
// Fields are public so free-standing merge() (Chan et al.'s parallel
// combination formula) can combine two threads' partial moments without
// re-touching any sample.
struct OnlineMoments {
    std::int64_t n = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double x) {
        ++n;
        double delta = x - mean;
        mean += delta / static_cast<double>(n);
        m2 += delta * (x - mean);
    }

    double std_error() const {
        if (n < 2) return 0.0;
        double sample_var = m2 / static_cast<double>(n - 1);
        return std::sqrt(sample_var / static_cast<double>(n));
    }
};

OnlineMoments merge(const OnlineMoments& a, const OnlineMoments& b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    OnlineMoments out;
    out.n = a.n + b.n;
    double delta = b.mean - a.mean;
    out.mean = a.mean + delta * static_cast<double>(b.n) / static_cast<double>(out.n);
    out.m2 = a.m2 + b.m2 +
             delta * delta * static_cast<double>(a.n) * static_cast<double>(b.n) /
                 static_cast<double>(out.n);
    return out;
}

unsigned n_worker_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : n;
}

// Per-thread seed offset: a large odd constant (golden-ratio-derived, a
// common PRNG seed-spacing trick) so adjacent thread indices don't produce
// near-identical mt19937_64 initial states.
constexpr std::uint64_t kThreadSeedStride = 0x9E3779B97F4A7C15ULL;

}  // namespace

MCResult price_naive(double S, double K, double T, double r, double q, double sigma,
                      std::int64_t n_paths, OptionType option_type, std::uint64_t seed) {
    validate(n_paths);
    const double drift = (r - q - 0.5 * sigma * sigma) * T;
    const double diffusion_scale = sigma * std::sqrt(T);
    const double disc = std::exp(-r * T);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    OnlineMoments moments;
    for (std::int64_t i = 0; i < n_paths; ++i) {
        double z = normal(rng);
        double S_T = S * std::exp(drift + diffusion_scale * z);
        moments.add(disc * intrinsic(S_T, K, option_type));
    }
    return {moments.mean, moments.std_error()};
}

MCResult price_antithetic(double S, double K, double T, double r, double q, double sigma,
                           std::int64_t n_paths, OptionType option_type, std::uint64_t seed) {
    validate(n_paths);
    const std::int64_t n_pairs = n_paths / 2;
    if (n_pairs < 1) {
        throw std::invalid_argument("n_paths must be >= 2 to form at least one antithetic pair");
    }
    const double drift = (r - q - 0.5 * sigma * sigma) * T;
    const double diffusion_scale = sigma * std::sqrt(T);
    const double disc = std::exp(-r * T);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    OnlineMoments moments;
    for (std::int64_t i = 0; i < n_pairs; ++i) {
        double z = normal(rng);
        double diffusion = diffusion_scale * z;
        double S_up = S * std::exp(drift + diffusion);
        double S_down = S * std::exp(drift - diffusion);
        double pair_payoff =
            0.5 * (intrinsic(S_up, K, option_type) + intrinsic(S_down, K, option_type));
        moments.add(disc * pair_payoff);
    }
    return {moments.mean, moments.std_error()};
}

MCResult price_control_variate(double S, double K, double T, double r, double q, double sigma,
                                std::int64_t n_paths, OptionType option_type, std::uint64_t seed) {
    validate(n_paths);
    const double drift = (r - q - 0.5 * sigma * sigma) * T;
    const double diffusion_scale = sigma * std::sqrt(T);
    const double disc = std::exp(-r * T);
    const double E_X = S * std::exp(-q * T);

    // Pass 1: accumulate the sums needed for b = Cov(Y, X) / Var(X). Two
    // fresh engine+distribution pairs (one per pass) guarantee the second
    // pass replays the identical draw sequence rather than storing it.
    std::mt19937_64 rng1(seed);
    std::normal_distribution<double> normal1(0.0, 1.0);

    double sumY = 0.0, sumX = 0.0, sumXY = 0.0, sumX2 = 0.0;
    for (std::int64_t i = 0; i < n_paths; ++i) {
        double z = normal1(rng1);
        double S_T = S * std::exp(drift + diffusion_scale * z);
        double Y = disc * intrinsic(S_T, K, option_type);
        double X = disc * S_T;
        sumY += Y;
        sumX += X;
        sumXY += X * Y;
        sumX2 += X * X;
    }
    const double n = static_cast<double>(n_paths);
    const double meanY = sumY / n;
    const double meanX = sumX / n;
    const double cov = (sumXY - n * meanX * meanY) / (n - 1.0);
    const double var = (sumX2 - n * meanX * meanX) / (n - 1.0);
    const double b = cov / var;

    // Pass 2: replay the same sequence, applying the control adjustment.
    std::mt19937_64 rng2(seed);
    std::normal_distribution<double> normal2(0.0, 1.0);

    OnlineMoments moments;
    for (std::int64_t i = 0; i < n_paths; ++i) {
        double z = normal2(rng2);
        double S_T = S * std::exp(drift + diffusion_scale * z);
        double Y = disc * intrinsic(S_T, K, option_type);
        double X = disc * S_T;
        moments.add(Y - b * (X - E_X));
    }
    return {moments.mean, moments.std_error()};
}

MCResult price_naive_parallel(double S, double K, double T, double r, double q, double sigma,
                               std::int64_t n_paths, OptionType option_type, std::uint64_t seed) {
    validate(n_paths);
    const double drift = (r - q - 0.5 * sigma * sigma) * T;
    const double diffusion_scale = sigma * std::sqrt(T);
    const double disc = std::exp(-r * T);

    const unsigned n_threads = n_worker_threads();
    std::vector<OnlineMoments> partials(n_threads);
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    for (unsigned t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, t]() {
            const std::int64_t start = (n_paths * static_cast<std::int64_t>(t)) / n_threads;
            const std::int64_t end = (n_paths * static_cast<std::int64_t>(t + 1)) / n_threads;

            std::mt19937_64 rng(seed + static_cast<std::uint64_t>(t) * kThreadSeedStride);
            std::normal_distribution<double> normal(0.0, 1.0);

            OnlineMoments local;
            for (std::int64_t i = start; i < end; ++i) {
                double z = normal(rng);
                double S_T = S * std::exp(drift + diffusion_scale * z);
                local.add(disc * intrinsic(S_T, K, option_type));
            }
            partials[t] = local;
        });
    }
    for (auto& worker : workers) worker.join();

    OnlineMoments combined;
    for (const auto& partial : partials) combined = merge(combined, partial);
    return {combined.mean, combined.std_error()};
}

MCResult price_antithetic_parallel(double S, double K, double T, double r, double q, double sigma,
                                    std::int64_t n_paths, OptionType option_type,
                                    std::uint64_t seed) {
    validate(n_paths);
    const std::int64_t n_pairs = n_paths / 2;
    if (n_pairs < 1) {
        throw std::invalid_argument("n_paths must be >= 2 to form at least one antithetic pair");
    }
    const double drift = (r - q - 0.5 * sigma * sigma) * T;
    const double diffusion_scale = sigma * std::sqrt(T);
    const double disc = std::exp(-r * T);

    const unsigned n_threads = n_worker_threads();
    std::vector<OnlineMoments> partials(n_threads);
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    for (unsigned t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, t]() {
            const std::int64_t start = (n_pairs * static_cast<std::int64_t>(t)) / n_threads;
            const std::int64_t end = (n_pairs * static_cast<std::int64_t>(t + 1)) / n_threads;

            std::mt19937_64 rng(seed + static_cast<std::uint64_t>(t) * kThreadSeedStride);
            std::normal_distribution<double> normal(0.0, 1.0);

            OnlineMoments local;
            for (std::int64_t i = start; i < end; ++i) {
                double z = normal(rng);
                double diffusion = diffusion_scale * z;
                double S_up = S * std::exp(drift + diffusion);
                double S_down = S * std::exp(drift - diffusion);
                double pair_payoff =
                    0.5 * (intrinsic(S_up, K, option_type) + intrinsic(S_down, K, option_type));
                local.add(disc * pair_payoff);
            }
            partials[t] = local;
        });
    }
    for (auto& worker : workers) worker.join();

    OnlineMoments combined;
    for (const auto& partial : partials) combined = merge(combined, partial);
    return {combined.mean, combined.std_error()};
}

}  // namespace pricing
