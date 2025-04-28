#ifndef SAMPLE_CONTROLS_HOST_HPP
#define SAMPLE_CONTROLS_HOST_HPP

#include "../include/kokkos_types.hpp"
#include <Kokkos_Core.hpp>

// Fills a Host matrix with random controls (standard normal samples)
void sample_controls_host_host(
    int total_samples,
    HostMatrix& random_controls_out
);

#endif // SAMPLE_CONTROLS_HOST_HPP
