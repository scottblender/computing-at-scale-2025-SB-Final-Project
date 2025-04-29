#ifndef COMPUTE_TRANSFORM_MATRIX_HPP
#define COMPUTE_TRANSFORM_MATRIX_HPP

#include <Kokkos_Core.hpp>

void compute_transform_matrix(
    const Kokkos::View<double**>& transform_out
);

#endif // COMPUTE_TRANSFORM_MATRIX_HPP
