cmake_minimum_required(VERSION 3.28.3)
project(final_project CXX)

# --------------------------
# C++ Standard
# --------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------
# Dependency Paths (match install.sh)
# --------------------------
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/deps/kokkos/install")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/deps/catch2/install")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/deps/eigen/install")

# --------------------------
# Find Dependencies
# --------------------------
find_package(Kokkos REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Catch2 3 REQUIRED)

# --------------------------
# Sigma Point Library (CPU version — FULL .cpp + .hpp like before)
# --------------------------
add_library(sigma_points_lib
    CPU/src/sigma_points_kokkos.cpp
    CPU/src/sigma_propagation.cpp
    CPU/src/rv2mee.cpp
    CPU/src/mee2rv.cpp
    CPU/src/odefunc.cpp
    CPU/src/l1_dot_2B_propul.cpp
    CPU/src/lm_dot_2B_propul.cpp
)
target_include_directories(sigma_points_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(sigma_points_lib
    PUBLIC
    Kokkos::kokkos
    Eigen3::Eigen
)

# --------------------------
# Sigma Point Library (GPU version — mix of .cpp + header-only)
# --------------------------
add_library(sigma_points_gpu_lib
    GPU/src/sigma_propagation_gpu.cpp
    GPU/src/sigma_points_kokkos_gpu.cpp
    GPU/src/sample_controls_host.cpp
    GPU/src/compute_transform_matrix.cpp
)
target_include_directories(sigma_points_gpu_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(sigma_points_gpu_lib
    PUBLIC
    Kokkos::kokkos
    Eigen3::Eigen
)

# --------------------------
# Unit Tests: Sigma Points (CPU)
# --------------------------
add_executable(test_sigma_points
    CPU/src/main.cpp
    CPU/tests/test_sigma_points.cpp
)
target_link_libraries(test_sigma_points
    PRIVATE
    sigma_points_lib
    Catch2::Catch2
)

# --------------------------
# Unit Tests: Sigma Propagation (CPU)
# --------------------------
add_executable(test_sigma_propagation
    CPU/src/main.cpp
    CPU/tests/test_sigma_propagation.cpp
)
target_link_libraries(test_sigma_propagation
    PRIVATE
    sigma_points_lib
    Catch2::Catch2
)

# --------------------------
# Additional Test: Sigma Propagation CSV Validation (CPU)
# --------------------------
add_executable(test_sigma_propagation_cpu
    CPU/src/main.cpp
    CPU/tests/test_sigma_propagation_cpu.cpp
)
target_link_libraries(test_sigma_propagation_cpu
    PRIVATE
    sigma_points_lib
    Catch2::Catch2
)

# --------------------------
# GPU Tests: Sigma Propagation (GPU)
# --------------------------
add_executable(test_sigma_propagation_gpu
    GPU/src/main.cpp
    GPU/tests/test_sigma_propagation_gpu.cpp
)
target_link_libraries(test_sigma_propagation_gpu
    PRIVATE
    sigma_points_gpu_lib
    Catch2::Catch2
)

# --------------------------
# Benchmark executable: Timing Serial/OpenMP/CUDA (NEW)
# --------------------------
add_executable(runtime_vs_timesteps_benchmark
    GPU/src/main.cpp  # uses new main.cpp you wrote with --benchmark option
)
target_link_libraries(runtime_vs_timesteps_benchmark
    PRIVATE
    sigma_points_gpu_lib
    Catch2::Catch2
)
