#ifndef SIGMA_POINTS_KOKKOS_GPU_HPP
#define SIGMA_POINTS_KOKKOS_GPU_HPP

#include <Kokkos_Core.hpp>
#include "../include/kokkos_types.hpp"

// Function to generate sigma points using Kokkos parallelism
void generate_sigma_points_kokkos(
    int nsd,
    double alpha,
    double beta,
    double kappa,
    const double* P_pos_flat,
    const double* P_vel_flat,
    double P_mass,
    const Kokkos::View<int*, MEMORY_SPACE>& time_steps,
    const View3D& r_bundles,
    const View3D& v_bundles,
    const View2D& m_bundles,
    const View4D& sigmas_out
);

#endif // SIGMA_POINTS_KOKKOS_GPU_HPP
