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
    sigma_points_kokkos.cpp
)
target_include_directories(sigma_points_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(sigma_points_lib
    PUBLIC
    Kokkos::kokkos
    Eigen3::Eigen
)

# --------------------------
# Unit Tests
# --------------------------
add_executable(test_sigma_points
    main.cpp
    test_sigma_points.cpp
)
target_link_libraries(test_sigma_points
    PRIVATE
    sigma_points_lib
    Catch2::Catch2
)
