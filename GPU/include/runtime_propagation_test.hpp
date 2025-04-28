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
#include "../include/rv2mee_gpu.hpp"

// ==================================================
// Timing utility to run propagation test
// ==================================================

inline double run_propagation_test(int num_steps, const PropagationSettings& settings) {
    double elapsed = 0.0;

    {
        const int num_bundles = 1;
        const int nsd = 7;
        const int num_sigma = 2 * nsd + 1;  // 15 sigma points
        const int num_intervals = num_steps - 1; // intervals = steps - 1

        if (num_steps < 2) {
            std::cerr << "[ERROR] num_steps must be >= 2!\n";
            return -1.0;
        }

        // Load initial_bundle_32.csv
        auto initial_data_vec = load_csv("initial_bundle_32.csv", 16);
        const int total_rows = static_cast<int>(initial_data_vec.size());
        if (num_steps > total_rows) {
            std::cerr << "[ERROR] Not enough rows in initial_bundle_32.csv for requested num_steps.\n";
            return -1.0;
        }

        // Fill initial data
        Eigen::MatrixXd initial_data(num_steps, 16);
        for (int i = 0; i < num_steps; ++i) {
            for (int j = 0; j < 16; ++j) {
                initial_data(i, j) = initial_data_vec[i][j];
            }
        }

        Kokkos::View<double***> r_bundles("r_bundles", num_bundles, num_steps, 3);
        Kokkos::View<double***> v_bundles("v_bundles", num_bundles, num_steps, 3);
        Kokkos::View<double**> m_bundles("m_bundles", num_bundles, num_steps);
        Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, nsd, num_bundles);

        std::vector<double> time_vec(num_steps);

        for (int step = 0; step < num_steps; ++step) {
            for (int k = 0; k < 3; ++k) {
                r_bundles(0, step, k) = initial_data(step, 1 + k);
                v_bundles(0, step, k) = initial_data(step, 4 + k);
            }
            m_bundles(0, step) = initial_data(step, 7);
            for (int k = 0; k < 7; ++k) {
                new_lam_bundles(step, k, 0) = initial_data(step, 8 + k);
            }
            time_vec[step] = initial_data(step, 0);
        }

        // Sigma points
        Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, nsd, num_steps);

        Kokkos::View<int*> time_steps_view("time_steps_view", num_steps);
        auto time_steps_host = Kokkos::create_mirror_view(time_steps_view);
        for (int i = 0; i < num_steps; ++i) {
            time_steps_host(i) = i;
        }
        Kokkos::deep_copy(time_steps_view, time_steps_host);

        double alpha = 1.7215, beta = 2.0, kappa = 3.0 - nsd;
        double P_mass = 0.0001;
        double P_pos_flat[9] = {0.01,0,0,0,0.01,0,0,0,0.01};
        double P_vel_flat[9] = {0.0001,0,0,0,0.0001,0,0,0,0.0001};

        generate_sigma_points_kokkos(
            nsd, alpha, beta, kappa,
            P_pos_flat, P_vel_flat, P_mass,
            time_steps_view, r_bundles, v_bundles, m_bundles,
            sigmas_combined
        );

        // Random controls
        const int num_random_samples = num_intervals * (settings.num_subintervals - 1);
        Kokkos::View<double**> random_controls("random_controls", num_random_samples, nsd);
        sample_controls_host(num_random_samples, random_controls);

        Kokkos::View<double**> transform("transform", nsd, nsd);
        compute_transform_matrix(transform);

        // Time and Weights
        Kokkos::View<double*> time_view("time_view", num_steps);
        auto time_host = Kokkos::create_mirror_view(time_view);
        for (int i = 0; i < num_steps; ++i) {
            time_host(i) = time_vec[i];
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

        // Trajectories (always fixed to 200 points total per full trajectory)
        int num_storage_steps = settings.num_eval_per_step;  // ALWAYS 200
        Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, num_storage_steps, 8);

        // === Start timing ===
        Kokkos::Timer timer;

        propagate_sigma_trajectories(
            sigmas_combined, new_lam_bundles,
            time_view, Wm_view, Wc_view,
            random_controls, transform,
            settings,
            trajectories_out
        );

        Kokkos::fence();
        elapsed = timer.seconds();
    } // All Views destroyed safely here

    return elapsed;
}

#endif // RUNTIME_PROPAGATION_TEST_HPP
