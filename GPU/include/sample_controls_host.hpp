#ifndef SAMPLE_CONTROLS_HOST_HPP
#define SAMPLE_CONTROLS_HOST_HPP

#include <Kokkos_Core.hpp>

// Conditionally define memory space based on CUDA availability
#ifdef KOKKOS_ENABLE_CUDA
    #define MEMORY_SPACE Kokkos::CudaSpace
#else
    #define MEMORY_SPACE Kokkos::HostSpace
#endif

// Function to fill a Host matrix with random controls (standard normal samples)
void sample_controls_host_host(
    int total_samples,
    Kokkos::View<double**, MEMORY_SPACE>& random_controls_out
);

#endif // SAMPLE_CONTROLS_HOST_HPP
