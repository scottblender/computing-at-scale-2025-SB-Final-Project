#ifndef SIGMA_POINTS_KOKKOS_GPU_HPP
#define SIGMA_POINTS_KOKKOS_GPU_HPP

#include <Kokkos_Core.hpp>

// Typedefs for convenience
using View3D = Kokkos::View<double***>;
using View2D = Kokkos::View<double**>;
using View4D = Kokkos::View<double****>;
using ViewMatrixDevice = Kokkos::View<double**>; // Device-space L matrix

// Function to generate sigma points using Kokkos parallelism
void generate_sigma_points_kokkos(
    int nsd,
    double scaling_factor,
    const ViewMatrixDevice& L_device,
    const Kokkos::View<int*>& time_steps,
    const View3D& r_bundles,
    const View3D& v_bundles,
    const View2D& m_bundles,
    const View4D& sigmas_out
);

#endif // SIGMA_POINTS_KOKKOS_GPU_HPP
