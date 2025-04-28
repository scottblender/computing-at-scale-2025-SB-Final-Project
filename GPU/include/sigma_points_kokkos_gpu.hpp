#ifndef SIGMA_POINTS_KOKKOS_GPU_HPP
#define SIGMA_POINTS_KOKKOS_GPU_HPP

#include <Kokkos_Core.hpp>

using View3D = Kokkos::View<double***>;
using View2D = Kokkos::View<double**>;
using View4D = Kokkos::View<double****>;
using ViewMatrixHost = Kokkos::View<double**, Kokkos::HostSpace>;

void generate_sigma_points_kokkos(
    int nsd,
    double alpha,
    double beta,
    double kappa,
    const double* P_pos_flat, // 9 elements
    const double* P_vel_flat, // 9 elements
    double P_mass,
    const Kokkos::View<int*> time_steps, // now Kokkos View
    const View3D& r_bundles,
    const View3D& v_bundles,
    const View2D& m_bundles,
    const View4D& sigmas_out
);

#endif // SIGMA_POINTS_KOKKOS_GPU_HPP
