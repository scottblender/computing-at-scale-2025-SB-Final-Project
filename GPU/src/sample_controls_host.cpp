#include "../include/sample_controls_host.hpp"
#include <random>

// Fill a HostMatrix with random samples
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
