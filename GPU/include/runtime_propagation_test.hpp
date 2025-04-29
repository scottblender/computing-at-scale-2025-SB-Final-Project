#ifndef RUNTIME_PROPAGATION_TEST_HPP
#define RUNTIME_PROPAGATION_TEST_HPP

#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
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
#include "../include/kokkos_types.hpp"

// Conditionally define memory space based on CUDA availability
#ifdef KOKKOS_ENABLE_CUDA
    // If CUDA is enabled, use CudaSpace
    #define MEMORY_SPACE Kokkos::CudaSpace
#else
    // Otherwise, use HostSpace (default for serial runs)
    #define MEMORY_SPACE Kokkos::HostSpace
#endif

inline double run_propagation_test(int num_steps, const PropagationSettings& settings) {
    double elapsed = 0.0;

    const int num_bundles = 1;
    const int nsd = 7;
    const int num_sigma = 2 * nsd + 1;

    if (num_steps < 2) {
        std::cerr << "[ERROR] num_steps must be >= 2!\n";
        return -1.0;
    }

    auto initial_data_vec = load_csv("initial_bundle_32.csv", 16);
    if (num_steps > static_cast<int>(initial_data_vec.size())) {
        std::cerr << "[ERROR] Not enough rows in initial_bundle_32.csv for requested num_steps.\n";
        return -1.0;
    }

    Eigen::MatrixXd initial_data(num_steps, 16);
    for (int i = 0; i < num_steps; ++i)
        for (int j = 0; j < 16; ++j)
            initial_data(i, j) = initial_data_vec[i][j];

    // Kokkos views for bundle data
    Kokkos::View<double***, MEMORY_SPACE> r_bundles("r_bundles", num_bundles, 2, 3);
    Kokkos::View<double***, MEMORY_SPACE> v_bundles("v_bundles", num_bundles, 2, 3);
    Kokkos::View<double**, MEMORY_SPACE> m_bundles("m_bundles", num_bundles, 2);
    Kokkos::View<double***, MEMORY_SPACE> new_lam_bundles("new_lam_bundles", 2, nsd, num_bundles);
    Kokkos::View<int*, MEMORY_SPACE> time_steps_view("time_steps_view", 2);  // Changed to int* for consistency with the function signature
    Kokkos::View<double*, MEMORY_SPACE> time_view("time_view", 2);

    Kokkos::View<double*, MEMORY_SPACE> Wm_view("Wm", num_sigma);
    Kokkos::View<double*, MEMORY_SPACE> Wc_view("Wc", num_sigma);
    auto Wm_host = Kokkos::create_mirror_view(Wm_view);
    auto Wc_host = Kokkos::create_mirror_view(Wc_view);

    double alpha = 1.7215;
    for (int i = 0; i < num_sigma; ++i) {
        Wm_host(i) = 1.0 / (2 * (nsd + alpha));
        Wc_host(i) = Wm_host(i);
    }
    Kokkos::deep_copy(Wm_view, Wm_host);
    Kokkos::deep_copy(Wc_view, Wc_host);

    const int num_subintervals = settings.num_subintervals;
    const int num_random_samples_per_interval = num_subintervals - 1;
    const int total_random_samples = (num_steps - 1) * num_random_samples_per_interval;

    // Declare the device view using the conditionally defined MEMORY_SPACE
    Kokkos::View<double**, MEMORY_SPACE> random_controls("random_controls", total_random_samples, nsd);

    // Declare the host view
    Kokkos::View<double**, Kokkos::HostSpace> random_controls_host("random_controls_host", total_random_samples, nsd);

    // Fill host matrix
    sample_controls_host_host(total_random_samples, random_controls_host);

    // Copy data from host to device
    Kokkos::deep_copy(random_controls, random_controls_host);  // Ensure layouts match
    
    // Create transform matrix view
    Kokkos::View<double**, MEMORY_SPACE> transform("transform", nsd, nsd);
    compute_transform_matrix(transform);

    int num_storage_steps = settings.num_eval_per_step;
    Kokkos::View<double****, MEMORY_SPACE> trajectories_out("trajectories_out", num_bundles, num_sigma, num_storage_steps, 8);

    Kokkos::Timer timer;
    int random_sample_idx = 0;

    for (int j = 0; j < num_steps - 1; ++j) {
        auto r_host = Kokkos::create_mirror_view(r_bundles);
        auto v_host = Kokkos::create_mirror_view(v_bundles);
        auto m_host = Kokkos::create_mirror_view(m_bundles);
        auto lam_host = Kokkos::create_mirror_view(new_lam_bundles);
        auto time_steps_host = Kokkos::create_mirror_view(time_steps_view);
        auto time_host = Kokkos::create_mirror_view(time_view);

        for (int step = 0; step < 2; ++step) {
            for (int k = 0; k < 3; ++k) {
                r_host(0, step, k) = initial_data(j + step, 1 + k);
                v_host(0, step, k) = initial_data(j + step, 4 + k);
            }
            m_host(0, step) = initial_data(j + step, 7);
            for (int k = 0; k < 7; ++k)
                lam_host(step, k, 0) = initial_data(j + step, 8 + k);
            time_steps_host(step) = step;
            time_host(step) = initial_data(j + step, 0);
        }

        Kokkos::deep_copy(r_bundles, r_host);
        Kokkos::deep_copy(v_bundles, v_host);
        Kokkos::deep_copy(m_bundles, m_host);
        Kokkos::deep_copy(new_lam_bundles, lam_host);
        Kokkos::deep_copy(time_steps_view, time_steps_host);
        Kokkos::deep_copy(time_view, time_host);

        Device4D sigmas_combined("sigmas_combined", num_bundles, num_sigma, nsd, 2);

        double P_mass = 0.0001;
        double P_pos_flat[9] = {0.01,0,0,0,0.01,0,0,0,0.01};
        double P_vel_flat[9] = {0.0001,0,0,0,0.0001,0,0,0,0.0001};

        generate_sigma_points_kokkos(
            nsd, alpha, 2.0, 3.0 - nsd,
            P_pos_flat, P_vel_flat, P_mass,
            time_steps_view, r_bundles, v_bundles, m_bundles, sigmas_combined
        );

        auto random_controls_sub = Kokkos::subview(
            random_controls,
            Kokkos::pair<int,int>(random_sample_idx, random_sample_idx + num_random_samples_per_interval),
            Kokkos::ALL()
        );

        propagate_sigma_trajectories(
            sigmas_combined, new_lam_bundles,
            time_view, Wm_view, Wc_view,
            random_controls_sub, transform,
            settings,
            trajectories_out
        );

        random_sample_idx += num_random_samples_per_interval;
    }

    Kokkos::fence();
    elapsed = timer.seconds();

    return elapsed;
}

#endif // RUNTIME_PROPAGATION_TEST_HPP
