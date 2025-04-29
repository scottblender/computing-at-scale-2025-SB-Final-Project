#ifndef SIGMA_PROPAGATION_GPU_HPP
#define SIGMA_PROPAGATION_GPU_HPP

#include <Kokkos_Core.hpp>
#include "propagation_settings.hpp"

// Define the memory space based on whether CUDA is enabled or not
#ifdef KOKKOS_ENABLE_CUDA
    #define MEMORY_SPACE Kokkos::CudaSpace
#else
    #define MEMORY_SPACE Kokkos::HostSpace
#endif

// Define Kokkos View types based on MEMORY_SPACE
using View4D = Kokkos::View<double****, MEMORY_SPACE>;
using View3D = Kokkos::View<double***, MEMORY_SPACE>;
using View2D = Kokkos::View<double**, MEMORY_SPACE>;
using View1D = Kokkos::View<double*, MEMORY_SPACE>;

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
