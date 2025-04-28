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
// Timing utility to run correct benchmark
// ==================================================

inline double run_propagation_test(int num_steps, const PropagationSettings& settings) {
    double elapsed = 0.0;

    const int nsd = 7;
    const int num_sigma = 2 * nsd + 1; // 15 sigma points
    const int num_bundles = 1;         // still one bundle per interval

    // 1. Load the initial data CSV
    auto initial_data_vec = load_csv("initial_bundle_32.csv", 16);
    const int total_rows = initial_data_vec.size();

    if (num_steps + 1 > total_rows) {
        throw std::runtime_error("[run_propagation_test] Not enough rows in initial_bundle_32.csv for requested number of steps!");
    }

    // 2. Load the sigma weights (equal weights for now)
    Kokkos::View<double*> Wm_view("Wm_view", num_sigma);
    Kokkos::View<double*> Wc_view("Wc_view", num_sigma);
    auto Wm_host = Kokkos::create_mirror_view(Wm_view);
    auto Wc_host = Kokkos::create_mirror_view(Wc_view);

    double alpha = 1.7215;
    for (int i = 0; i < num_sigma; ++i) {
        Wm_host(i) = 1.0 / (2.0 * (nsd + alpha));
        Wc_host(i) = Wm_host(i);
    }
    Kokkos::deep_copy(Wm_view, Wm_host);
    Kokkos::deep_copy(Wc_view, Wc_host);

    // 3. Random control matrix and transform matrix
    const int num_random_samples = (settings.num_subintervals - 1);
    Kokkos::View<double**> random_controls("random_controls", num_random_samples, nsd);
    Kokkos::View<double**> transform("transform", nsd, nsd);
    sample_controls_host(num_random_samples, random_controls);
    compute_transform_matrix(transform);

    // 4. Start timer
    Kokkos::Timer timer;

    // 5. Loop over each interval [t_i, t_{i+1}]
    for (int i = 0; i < num_steps; ++i) {
        // 5.1 Setup time bounds
        double t0 = initial_data_vec[i][0];
        double t1 = initial_data_vec[i+1][0];

        // 5.2 Setup initial state
        double r[3], v[3];
        for (int k = 0; k < 3; ++k) {
            r[k] = initial_data_vec[i][1 + k];
            v[k] = initial_data_vec[i][4 + k];
        }
        double mass = initial_data_vec[i][7];

        double mee[6];
        rv2mee(r, v, settings.mu, mee);

        double state[14];
        for (int k = 0; k < 6; ++k) state[k] = mee[k];
        state[6] = mass;
        for (int k = 0; k < 7; ++k) state[7 + k] = initial_data_vec[i][8 + k];

        // 5.3 Generate sigma points for THIS starting point
        Kokkos::View<double***> r_bundle("r_bundle", num_bundles, 1, 3);
        Kokkos::View<double***> v_bundle("v_bundle", num_bundles, 1, 3);
        Kokkos::View<double**> m_bundle("m_bundle", num_bundles, 1);
        Kokkos::View<double***> lam_bundle("lam_bundle", 1, nsd, num_bundles);

        // Initialize
        auto r_host = Kokkos::create_mirror_view(r_bundle);
        auto v_host = Kokkos::create_mirror_view(v_bundle);
        auto m_host = Kokkos::create_mirror_view(m_bundle);
        auto lam_host = Kokkos::create_mirror_view(lam_bundle);

        for (int k = 0; k < 3; ++k) {
            r_host(0, 0, k) = r[k];
            v_host(0, 0, k) = v[k];
        }
        m_host(0, 0) = mass;
        for (int k = 0; k < 7; ++k) {
            lam_host(0, k, 0) = initial_data_vec[i][8 + k];
        }

        Kokkos::deep_copy(r_bundle, r_host);
        Kokkos::deep_copy(v_bundle, v_host);
        Kokkos::deep_copy(m_bundle, m_host);
        Kokkos::deep_copy(lam_bundle, lam_host);

        Kokkos::View<int*> time_steps_view("time_steps", 1);
        auto time_steps_host = Kokkos::create_mirror_view(time_steps_view);
        time_steps_host(0) = 0;
        Kokkos::deep_copy(time_steps_view, time_steps_host);

        Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, nsd, 1);

        double P_pos_flat[9] = {0.01,0,0,0,0.01,0,0,0,0.01};
        double P_vel_flat[9] = {0.0001,0,0,0,0.0001,0,0,0,0.0001};
        double P_mass = 0.0001;

        generate_sigma_points_kokkos(
            nsd, alpha, 2.0, 3.0 - nsd,
            P_pos_flat, P_vel_flat, P_mass,
            time_steps_view,
            r_bundle, v_bundle, m_bundle,
            sigmas_combined
        );

        // 5.4 Propagate sigma points
        Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, settings.num_eval_per_step, 8);

        propagate_sigma_trajectories(
            sigmas_combined, lam_bundle,
            time_steps_view, Wm_view, Wc_view,
            random_controls, transform,
            settings,
            trajectories_out
        );
    }

    Kokkos::fence();
    elapsed = timer.seconds();
    return elapsed;
}

#endif // RUNTIME_PROPAGATION_TEST_HPP
