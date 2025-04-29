#include "../include/compute_transform_matrix.hpp"
#include <cmath>

void compute_transform_matrix(Kokkos::View<double**, MEMORY_SPACE>& transform_out) {
    auto mirror = Kokkos::create_mirror_view(transform_out);

    double P[7][7] = {0};
    for (int i = 0; i < 7; ++i)
        P[i][i] = 0.001;

    // Cholesky factor of P = L such that P = L * L^T (manual for diagonal matrix)
    for (int i = 0; i < 7; ++i) {
        mirror(i,i) = std::sqrt(P[i][i]);
        for (int j = 0; j < i; ++j) {
            mirror(i,j) = 0.0;
        }
    }

    Kokkos::deep_copy(transform_out, mirror);
}
