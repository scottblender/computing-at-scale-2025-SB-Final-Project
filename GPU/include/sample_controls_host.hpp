#ifndef SAMPLE_CONTROLS_HOST_HPP
#define SAMPLE_CONTROLS_HOST_HPP

#include <Kokkos_Core.hpp>

void sample_controls_host(
    int total_samples,
    Kokkos::View<double**>& random_controls_out
);

#endif // SAMPLE_CONTROLS_HOST_HPP
