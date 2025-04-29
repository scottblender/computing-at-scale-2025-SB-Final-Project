#ifndef COMPUTE_TRANSFORM_MATRIX_HPP
#define COMPUTE_TRANSFORM_MATRIX_HPP

#include <Kokkos_Core.hpp>

// Conditionally define memory space based on CUDA availability
#ifdef KOKKOS_ENABLE_CUDA
    #define MEMORY_SPACE Kokkos::CudaSpace
#else
    #define MEMORY_SPACE Kokkos::HostSpace
#endif

// Function to compute a transformation matrix, with flexible memory space
void compute_transform_matrix(
    Kokkos::View<double**, MEMORY_SPACE>& transform_out
);

#endif // COMPUTE_TRANSFORM_MATRIX_HPP
