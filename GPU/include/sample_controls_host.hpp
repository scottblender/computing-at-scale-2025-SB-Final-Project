#ifndef SAMPLE_CONTROLS_HOST_HPP
#define SAMPLE_CONTROLS_HOST_HPP

#include <Kokkos_Core.hpp>

// Define explicit types for device-side 2D matrix
using DeviceMatrix = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>;

void sample_controls_host(
    int total_samples,
    DeviceMatrix& random_controls_out
);

#endif // SAMPLE_CONTROLS_HOST_HPP
