#ifndef SIGMA_PROPAGATION_GPU_HPP
#define SIGMA_PROPAGATION_GPU_HPP

#include <Kokkos_Core.hpp>
#include "propagation_settings.hpp"

// Full GPU-compatible propagation function
void propagate_sigma_trajectories(
    const View4D& sigmas_combined,         // Input sigma points
    const View3D& new_lam_bundles,        // New lambda bundles
    const View1D& time,                   // Time vector
    const View1D& Wm,                     // Weights (mean)
    const View1D& Wc,                     // Weights (covariance)
    const View2D& random_controls,        // Random control inputs
    const View2D& transform,              // Transformation matrix
    const PropagationSettings& settings,  // Propagation settings
    View4D& trajectories_out              // Output trajectories
);

#endif // SIGMA_PROPAGATION_GPU_HPP
