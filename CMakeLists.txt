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
# Sigma Point Library
# --------------------------
add_library(sigma_points_lib
    CPU/sigma_points_kokkos.cpp
    CPU/sigma_propagation.cpp
    CPU/rv2mee.cpp
    CPU/mee2rv.cpp
    CPU/odefunc.cpp
    CPU/l1_dot_2B_propul.cpp
    CPU/lm_dot_2B_propul.cpp
)
target_include_directories(sigma_points_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(sigma_points_lib
    PUBLIC
    Kokkos::kokkos
    Eigen3::Eigen
)

# --------------------------
# Unit Tests: Sigma Points
# --------------------------
add_executable(test_sigma_points
    CPU/main.cpp
    CPU/test_sigma_points.cpp
)
target_link_libraries(test_sigma_points
    PRIVATE
    sigma_points_lib
    Catch2::Catch2
)

# --------------------------
# Unit Tests: Sigma Propagation
# --------------------------
add_executable(test_sigma_propagation
    CPU/main.cpp
    CPU/test_sigma_propagation.cpp
)
target_link_libraries(test_sigma_propagation
    PRIVATE
    sigma_points_lib
    Catch2::Catch2
)

# --------------------------
# Additional Test: Sigma Propagation CSV Validation
# --------------------------
add_executable(test_sigma_propagation_2
    CPU/main.cpp
    CPU/test_sigma_propagation_2.cpp
)
target_link_libraries(test_sigma_propagation_2
    PRIVATE
    sigma_points_lib
    Catch2::Catch2
)
