#ifndef SIGMA_PROPAGATION_GPU_HPP
#define SIGMA_PROPAGATION_GPU_HPP

#include <Kokkos_Core.hpp>
#include "propagation_settings.hpp"

// Define View types with explicit memory space for GPU compatibility
// Default execution space will be CUDA when compiled with CUDA enabled
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;

using View4D = Kokkos::View<double****, MemSpace>;
using View3D = Kokkos::View<double***, MemSpace>;
using View2D = Kokkos::View<double**, MemSpace>;
using View1D = Kokkos::View<double*, MemSpace>;

// Declaration for odefunc that will be called from device code
KOKKOS_INLINE_FUNCTION
void odefunc(const double* x, double* dx, double t, const PropagationSettings& settings);

// RK45 step function declaration 
KOKKOS_INLINE_FUNCTION
void rk45_step(
    void (*odefunc)(const double*, double*, double, const PropagationSettings&),
    const double* x0, double t0, double t1, int steps,
    const PropagationSettings& settings,
    double history[][14], double* time_out
);

// Main propagation function
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