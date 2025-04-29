#ifndef RUNTIME_PROPAGATION_TEST_HPP
#define RUNTIME_PROPAGATION_TEST_HPP

#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>

#include "../include/sigma_points_kokkos_gpu.hpp"
#include "../include/sigma_propagation_gpu.hpp"
#include "../include/csv_loader_gpu.hpp"
#include "../include/sample_controls_host.hpp"
#include "../include/compute_transform_matrix.hpp"
#include "../include/rv2mee_gpu.hpp"
#include "../include/kokkos_types.hpp"

#ifdef KOKKOS_ENABLE_CUDA
    #define MEMORY_SPACE Kokkos::CudaSpace
#else
    #define MEMORY_SPACE Kokkos::HostSpace
#endif

inline double run_propagation_test(int num_steps, const PropagationSettings& settings) {
    if (num_steps < 2) {
        std::cerr << "[ERROR] num_steps must be >= 2!\n";
        return -1.0;
    }

    Kokkos::Timer timer;

    auto all_data = load_csv("initial_bundles_32_33.csv", 17);
    std::unordered_map<int, std::vector<std::vector<double>>> bundle_rows;
    for (const auto& row : all_data) {
        int b = static_cast<int>(row[16]);
        bundle_rows[b].push_back(row);
    }

    const int num_bundles = static_cast<int>(bundle_rows.size());
    const int nsd = 7;
    const int num_sigma = 2 * nsd + 1;
    const int num_subintervals = settings.num_subintervals;
    const int num_random_samples_per_interval = num_subintervals - 1;
    const int total_random_samples = (num_steps - 1) * num_random_samples_per_interval;

    Kokkos::View<double***, MEMORY_SPACE> r_bundles("r_bundles", num_bundles, num_steps, 3);
    Kokkos::View<double***, MEMORY_SPACE> v_bundles("v_bundles", num_bundles, num_steps, 3);
    Kokkos::View<double**, MEMORY_SPACE> m_bundles("m_bundles", num_bundles, num_steps);
    Kokkos::View<double***, MEMORY_SPACE> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);
    Kokkos::View<int*, MEMORY_SPACE> time_steps_view("time_steps_view", num_steps);
    Kokkos::View<double*, MEMORY_SPACE> time_view("time_view", num_steps);

    auto r_host = Kokkos::create_mirror_view(r_bundles);
    auto v_host = Kokkos::create_mirror_view(v_bundles);
    auto m_host = Kokkos::create_mirror_view(m_bundles);
    auto lam_host = Kokkos::create_mirror_view(new_lam_bundles);
    auto time_steps_host = Kokkos::create_mirror_view(time_steps_view);
    auto time_host = Kokkos::create_mirror_view(time_view);

    int bundle_idx = 0;
    for (const auto& [bundle_id, rows] : bundle_rows) {
        if (rows.size() < num_steps) continue;
        for (int step = 0; step < num_steps; ++step) {
            const auto& row = rows[step];
            for (int k = 0; k < 3; ++k) {
                r_host(bundle_idx, step, k) = row[1 + k];
                v_host(bundle_idx, step, k) = row[4 + k];
            }
            m_host(bundle_idx, step) = row[7];
            for (int k = 0; k < 7; ++k)
                lam_host(step, k, bundle_idx) = row[8 + k];
            if (bundle_idx == 0) {
                time_host(step) = row[0];
                time_steps_host(step) = step;
            }
        }
        ++bundle_idx;
    }

    Kokkos::deep_copy(r_bundles, r_host);
    Kokkos::deep_copy(v_bundles, v_host);
    Kokkos::deep_copy(m_bundles, m_host);
    Kokkos::deep_copy(new_lam_bundles, lam_host);
    Kokkos::deep_copy(time_steps_view, time_steps_host);
    Kokkos::deep_copy(time_view, time_host);

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

    Kokkos::View<double**, Kokkos::HostSpace> random_controls_host("random_controls_host", total_random_samples, 7);
    sample_controls_host_host(total_random_samples, random_controls_host);

    Kokkos::View<double**, MEMORY_SPACE> random_controls_device("random_controls_device", total_random_samples, 7);

#ifdef KOKKOS_ENABLE_CUDA
    cudaMemcpy(random_controls_device.data(), random_controls_host.data(), total_random_samples * 7 * sizeof(double), cudaMemcpyHostToDevice);
#else
    Kokkos::deep_copy(random_controls_device, random_controls_host);
#endif

    Kokkos::View<double**, MEMORY_SPACE> transform("transform", 7, 7);
    compute_transform_matrix(transform);

    Kokkos::View<double****, MEMORY_SPACE> sigmas_combined("sigmas_combined", num_bundles, num_sigma, nsd, num_steps);

    double P_mass = 0.0001;
    double P_pos_flat[9] = {0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01};
    double P_vel_flat[9] = {0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.0001};

    generate_sigma_points_kokkos(
        nsd, alpha, 2.0, 3.0 - nsd,
        P_pos_flat, P_vel_flat, P_mass,
        time_steps_view, r_bundles, v_bundles, m_bundles,
        sigmas_combined
    );

    Kokkos::View<double****, MEMORY_SPACE> trajectories_out("trajectories_out", num_bundles, num_sigma, settings.num_eval_per_step, 8);

    propagate_sigma_trajectories(
        sigmas_combined, new_lam_bundles,
        time_view, Wm_view, Wc_view,
        random_controls_device, transform,
        settings,
        trajectories_out
    );

    Kokkos::fence();
    double total_elapsed = timer.seconds();
    std::cout << "[INFO] Total runtime: " << total_elapsed << " seconds\n";
    return total_elapsed;
}

#endif // RUNTIME_PROPAGATION_TEST_HPP
