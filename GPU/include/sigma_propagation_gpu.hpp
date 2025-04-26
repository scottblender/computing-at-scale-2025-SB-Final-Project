#ifndef SIGMA_PROPAGATION_GPU_HPP
#define SIGMA_PROPAGATION_GPU_HPP

#include <Kokkos_Core.hpp>
#include "propagation_settings.hpp"

using View4D = Kokkos::View<double****>;
using View3D = Kokkos::View<double***>;
using View2D = Kokkos::View<double**>;
using View1D = Kokkos::View<double*>;

// Full GPU-compatible propagation function
void propagate_sigma_trajectories(
    const View4D& sigmas_combined,
    const View3D& new_lam_bundles,
    const View1D& time,
    const View1D& Wm,
    const View1D& Wc,
    const View2D& random_controls, // shape: [samples, 7]
    const View2D& transform,       // shape: [7, 7]
    const PropagationSettings& settings,
    View4D& trajectories_out
);

#endif // SIGMA_PROPAGATION_GPU_HPP
