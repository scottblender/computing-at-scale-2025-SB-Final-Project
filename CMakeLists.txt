cmake_minimum_required(VERSION 3.28.3)
project(final_project CXX)

# --------------------------
# C++ Standard
# --------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------
# Kokkos
# --------------------------
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{HOME}/kokkos/install")
find_package(Kokkos REQUIRED)

# --------------------------
# Eigen
# --------------------------
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_SOURCE_DIR}/build-eigen/install")
find_package(Eigen3 REQUIRED)

# --------------------------
# Catch2
# --------------------------
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{HOME}/catch2/install")
find_package(Catch2 3 REQUIRED)

# --------------------------
# Sigma Point Library
# --------------------------
add_library(sigma_points_lib
    sigma_points_kokkos.cpp
)
target_include_directories(sigma_points_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(sigma_points_lib PUBLIC Kokkos::kokkos Eigen3::Eigen)

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
