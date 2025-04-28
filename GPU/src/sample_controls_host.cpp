#include "../include/sample_controls_host.hpp"
#include <random>

// Fill a Device matrix
void sample_controls_host(
    int total_samples,
    DeviceMatrix& random_controls_out
) {
    HostMatrix mirror = Kokkos::create_mirror_view(random_controls_out);

    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);

    for (int i = 0; i < total_samples; ++i) {
        for (int j = 0; j < 7; ++j) {
            mirror(i, j) = dist(gen);
        }
    }

    Kokkos::deep_copy(random_controls_out, mirror);
}

// Fill a Host matrix (no mirror needed)
void sample_controls_host_host(
    int total_samples,
    HostMatrix& random_controls_out
) {
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);

    for (int i = 0; i < total_samples; ++i) {
        for (int j = 0; j < 7; ++j) {
            random_controls_out(i, j) = dist(gen);
        }
    }
}
