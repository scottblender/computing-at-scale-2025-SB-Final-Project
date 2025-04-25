#include "sample_controls_host.hpp"
#include <random>

void sample_controls_host(
    int total_samples,
    Kokkos::View<double**>& random_controls_out
) {
    auto mirror = Kokkos::create_mirror_view(random_controls_out);
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);

    for (int i = 0; i < total_samples; ++i) {
        for (int j = 0; j < 7; ++j) {
            mirror(i,j) = dist(gen);
        }
    }
    Kokkos::deep_copy(random_controls_out, mirror);
}
