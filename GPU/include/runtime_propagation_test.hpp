#ifndef RUNTIME_PROPAGATION_TEST_HPP
#define RUNTIME_PROPAGATION_TEST_HPP

#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

#include "../include/sigma_points_kokkos_gpu.hpp"
#include "../include/sigma_propagation_gpu.hpp"
#include "../include/csv_loader_gpu.hpp"
#include "../include/sample_controls_host.hpp"
#include "../include/compute_transform_matrix.hpp"

// ==================================================
// Timing utility to run propagation test
// ==================================================

inline double run_propagation_test(int num_steps, const PropagationSettings& settings) {
    const int num_bundles = 1;
    const int nsd = 7;
    const int num_sigma = 2 * nsd + 1; // 15 sigma points

    // Dummy initializations
    Kokkos::View<double***> r_bundles("r_bundles", num_bundles, num_steps, 3);
    Kokkos::View<double***> v_bundles("v_bundles", num_bundles, num_steps, 3);
    Kokkos::View<double**> m_bundles("m_bundles", num_bundles, num_steps);
    Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, nsd, num_bundles);

    Kokkos::View<int*> time_steps("time_steps", num_steps);
    auto time_steps_host = Kokkos::create_mirror_view(time_steps);
    for (int i = 0; i < num_steps; ++i) {
        time_steps_host(i) = i;
    }
    Kokkos::deep_copy(time_steps, time_steps_host);

    double alpha = 1.7215, beta = 2.0, kappa = 3.0 - nsd;
    double P_mass = 0.0001;
    double P_pos_flat[9] = {0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01};
    double P_vel_flat[9] = {0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.0001};

    Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, nsd, num_steps);
    generate_sigma_points_kokkos(
        nsd, alpha, beta, kappa,
        P_pos_flat, P_vel_flat, P_mass,
        time_steps, r_bundles, v_bundles, m_bundles,
        sigmas_combined
    );

    // Random controls and transform
    const int num_random_samples = (num_steps - 1) * (settings.num_subintervals - 1);
    Kokkos::View<double**> random_controls("random_controls", num_random_samples, nsd);
    Kokkos::View<double**> transform("transform", nsd, nsd);
    sample_controls_host(num_random_samples, random_controls);
    compute_transform_matrix(transform);

    // Time and weights
    Kokkos::View<double*> time_view("time_view", num_steps);
    auto time_host = Kokkos::create_mirror_view(time_view);
    for (int i = 0; i < num_steps; ++i) {
        time_host(i) = static_cast<double>(i);
    }
    Kokkos::deep_copy(time_view, time_host);

    Kokkos::View<double*> Wm_view("Wm", num_sigma);
    Kokkos::View<double*> Wc_view("Wc", num_sigma);
    auto Wm_host = Kokkos::create_mirror_view(Wm_view);
    auto Wc_host = Kokkos::create_mirror_view(Wc_view);
    for (int i = 0; i < num_sigma; ++i) {
        Wm_host(i) = 1.0 / (2 * (nsd + alpha));
        Wc_host(i) = Wm_host(i);
    }
    Kokkos::deep_copy(Wm_view, Wm_host);
    Kokkos::deep_copy(Wc_view, Wc_host);

    Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, settings.num_eval_per_step, 8);

    // === Start timer ===
    Kokkos::Timer timer;

    propagate_sigma_trajectories(
        sigmas_combined, new_lam_bundles,
        time_view, Wm_view, Wc_view,
        random_controls, transform,
        settings,
        trajectories_out
    );

    Kokkos::fence();
    double elapsed = timer.seconds();
    return elapsed;
}

#endif // RUNTIME_PROPAGATION_TEST_HPP
