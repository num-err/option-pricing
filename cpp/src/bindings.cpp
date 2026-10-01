// pybind11 bindings exposing the C++ Monte Carlo kernels as the `pricing_cpp`
// Python extension module, so benchmarks/bench_monte_carlo.py can call the
// C++ implementation and the numpy implementation from the same process.
#include <pybind11/pybind11.h>

#include "pricing/binomial.hpp"
#include "pricing/black_scholes.hpp"
#include "pricing/implied_vol.hpp"
#include "pricing/monte_carlo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pricing_cpp, m) {
    m.doc() = "C++ pricing kernels (performance counterpart to the pricing.* Python package)";

    py::enum_<pricing::OptionType>(m, "OptionType")
        .value("Call", pricing::OptionType::Call)
        .value("Put", pricing::OptionType::Put);

    py::class_<pricing::Greeks>(m, "Greeks")
        .def_readonly("price", &pricing::Greeks::price)
        .def_readonly("delta", &pricing::Greeks::delta)
        .def_readonly("gamma", &pricing::Greeks::gamma)
        .def_readonly("vega", &pricing::Greeks::vega)
        .def_readonly("theta", &pricing::Greeks::theta)
        .def_readonly("rho", &pricing::Greeks::rho);

    m.def("price", &pricing::price, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("q"), py::arg("sigma"), py::arg("option_type"),
          "Closed-form Black-Scholes-Merton price.");
    m.def("delta", &pricing::delta, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("q"), py::arg("sigma"), py::arg("option_type"));
    m.def("gamma", &pricing::gamma, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("q"), py::arg("sigma"));
    m.def("vega", &pricing::vega, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("q"), py::arg("sigma"));
    m.def("theta", &pricing::theta, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("q"), py::arg("sigma"), py::arg("option_type"));
    m.def("rho", &pricing::rho, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("q"), py::arg("sigma"), py::arg("option_type"));
    m.def("greeks", &pricing::greeks, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("q"), py::arg("sigma"), py::arg("option_type"),
          "All of price/delta/gamma/vega/theta/rho, sharing one d1/d2 computation.");

    py::class_<pricing::MCResult>(m, "MCResult")
        .def_readonly("price", &pricing::MCResult::price)
        .def_readonly("std_error", &pricing::MCResult::std_error);

    m.def("price_naive", &pricing::price_naive, py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("n_paths"),
          py::arg("option_type"), py::arg("seed"),
          "Naive Monte Carlo: n_paths independent terminal draws.");

    m.def("price_antithetic", &pricing::price_antithetic, py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("n_paths"),
          py::arg("option_type"), py::arg("seed"),
          "Antithetic variates: n_paths total simulated draws.");

    m.def("price_control_variate", &pricing::price_control_variate, py::arg("S"), py::arg("K"),
          py::arg("T"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("n_paths"),
          py::arg("option_type"), py::arg("seed"),
          "Control variate: discounted terminal stock price as the control.");

    m.def("price_naive_parallel", &pricing::price_naive_parallel, py::arg("S"), py::arg("K"),
          py::arg("T"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("n_paths"),
          py::arg("option_type"), py::arg("seed"),
          "Naive Monte Carlo, split across std::thread::hardware_concurrency() threads.");

    m.def("price_antithetic_parallel", &pricing::price_antithetic_parallel, py::arg("S"),
          py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("n_paths"), py::arg("option_type"), py::arg("seed"),
          "Antithetic variates, split across std::thread::hardware_concurrency() threads.");

    m.def("binomial_price", &pricing::binomial_price, py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("N"), py::arg("option_type"),
          py::arg("american") = false,
          "CRR binomial tree price, European by default or American with american=True.");

    // Registered before the built-in std::invalid_argument -> ValueError
    // translator runs (pybind11 tries user-registered translators first, in
    // reverse registration order), so an ArbitrageViolation reaches Python
    // as that specific type rather than a generic ValueError -- mirroring
    // Python's `class ArbitrageViolation(ValueError)`.
    py::register_exception<pricing::ArbitrageViolation>(m, "ArbitrageViolation", PyExc_ValueError);

    py::class_<pricing::PriceBounds>(m, "PriceBounds")
        .def_readonly("lower", &pricing::PriceBounds::lower)
        .def_readonly("upper", &pricing::PriceBounds::upper);

    m.def("implied_vol_price_bounds", &pricing::implied_vol_price_bounds, py::arg("S"),
          py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("option_type"),
          "Model-free no-arbitrage (lower, upper) bounds for a European price.");

    m.def("implied_vol", &pricing::implied_vol, py::arg("price"), py::arg("S"), py::arg("K"),
          py::arg("T"), py::arg("r"), py::arg("q"), py::arg("option_type"),
          py::arg("sigma_low") = 1e-6, py::arg("sigma_high") = 5.0,
          "Invert black_scholes price for sigma via a from-scratch Brent's method.");
}
