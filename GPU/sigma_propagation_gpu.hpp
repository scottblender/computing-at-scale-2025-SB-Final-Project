#ifndef SIGMA_PROPAGATION_GPU_HPP
#define SIGMA_PROPAGATION_GPU_HPP

#include "sigma_propagation.hpp"
#include "rv2mee.hpp"
#include "mee2rv.hpp"
#include "odefunc.hpp"
#include "l1_dot_2B_propul.hpp"
#include "lm_dot_2B_propul.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <random>
#include <iostream>

// Function declarations
void sample_controls_device(
    const Kokkos::View<double***>& mean_controls,
    Kokkos::View<double**>& samples,
    int num_steps,
    int num_bundles,
    Kokkos::Random_XorShift64_Pool<>
);

struct RK45Integrator {
    Kokkos::View<double**> history;
    Kokkos::View<double*> time_vec;

    RK45Integrator(Kokkos::View<double**> h, Kokkos::View<double*> t) : history(h), time_vec(t) {}

    KOKKOS_FUNCTION void operator()(const int i) const;
};

void propagate_sigma_trajectories(
    const Kokkos::View<double****>& sigmas_combined,
    const Kokkos::View<double***>& new_lam_bundles,
    const std::vector<double>& time,
    const std::vector<double>& Wm,
    const std::vector<double>& Wc,
    const PropagationSettings& settings,
    Kokkos::View<double****>& trajectories_out
);

#endif // SIGMA_PROPAGATION_GPU_HPP