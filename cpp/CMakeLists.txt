cmake_minimum_required(VERSION 3.15)
project(pricing_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Core pricing kernels, independent of Python.
find_package(Threads REQUIRED)
add_library(pricing_core STATIC src/monte_carlo.cpp)
target_include_directories(pricing_core PUBLIC include)
target_link_libraries(pricing_core PUBLIC Threads::Threads)

# Python bindings (pybind11). Invoked through `uv run` so the `python3`
# resolved here is the project's own venv interpreter, which is where
# `uv add --group dev pybind11` installed pybind11's CMake package.
execute_process(
  COMMAND python3 -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pricing_cpp src/bindings.cpp)
target_link_libraries(pricing_cpp PRIVATE pricing_core)

# Standalone correctness test (no Python involved), run via ctest.
enable_testing()
add_executable(test_monte_carlo tests/test_monte_carlo.cpp)
target_link_libraries(test_monte_carlo PRIVATE pricing_core)
add_test(NAME monte_carlo COMMAND test_monte_carlo)
