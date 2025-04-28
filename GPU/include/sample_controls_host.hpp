#ifndef SAMPLE_CONTROLS_HOST_HPP
#define SAMPLE_CONTROLS_HOST_HPP

#include "../include/kokkos_types.hpp"

#include <Kokkos_Core.hpp>


// For device memory (real GPU benchmark)
void sample_controls_host(
    int total_samples,
    DeviceMatrix& random_controls_out
);

// For host memory (tests like Catch2)
void sample_controls_host_host(
    int total_samples,
    HostMatrix& random_controls_out
);

#endif // SAMPLE_CONTROLS_HOST_HPP
