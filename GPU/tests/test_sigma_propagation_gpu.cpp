#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>

#include "../include/csv_loader_gpu.hpp"
#include "../include/sigma_points_kokkos_gpu.hpp"
#include "../include/sigma_propagation_gpu.hpp"
#include "../include/sample_controls_host.hpp"
#include "../include/compute_transform_matrix.hpp"

// Conditionally define memory space based on CUDA availability
#ifdef KOKKOS_ENABLE_CUDA
    #define MEMORY_SPACE Kokkos::CudaSpace
#else
    #define MEMORY_SPACE Kokkos::HostSpace
#endif

TEST_CASE("Check propagated values for bundle=32, sigma=0 for single interval [GPU-compatible]", "[propagation]") {
    // Load initial CSV data
    auto all_data = load_csv("initial_bundles_32_33.csv", 16);
    std::vector<std::vector<double>> initial_data_vec;
    for (const auto& row : all_data) {
        if (static_cast<int>(row[15]) == 32)  
            initial_data_vec.push_back(row);
    }
    const int num_rows = static_cast<int>(initial_data_vec.size());
    const int num_cols = static_cast<int>(initial_data_vec[0].size());

    Eigen::MatrixXd initial_data(num_rows, num_cols);
    for (int i = 0; i < num_rows; ++i)
        for (int j = 0; j < num_cols; ++j)
            initial_data(i, j) = initial_data_vec[i][j];

    // Load sigma weights
    std::vector<double> Wm, Wc;
    load_weights("sigma_weights.csv", Wm, Wc);

    const int num_sigma = static_cast<int>(Wm.size());
    const int num_steps = 2;  // For single interval
    const int num_bundles = 1;
    const int nsd = 7;
    double alpha = 1.7215;
    double beta = 2.0;
    double kappa = 3.0 - nsd;
    
    // Kokkos views for bundle data
    Kokkos::View<double***, MEMORY_SPACE> r_bundles("r_bundles", num_bundles, num_steps, 3);
    Kokkos::View<double***, MEMORY_SPACE> v_bundles("v_bundles", num_bundles, num_steps, 3);
    Kokkos::View<double**, MEMORY_SPACE> m_bundles("m_bundles", num_bundles, num_steps);
    Kokkos::View<double***, MEMORY_SPACE> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);
    std::vector<double> time(num_steps);

    // Fill Kokkos views with initial data
    for (int step = 0; step < num_steps; ++step) {
        for (int k = 0; k < 3; ++k) {
            r_bundles(0, step, k) = initial_data(step, 1 + k);
            v_bundles(0, step, k) = initial_data(step, 4 + k);
        }
        m_bundles(0, step) = initial_data(step, 7);
        for (int k = 0; k < 7; ++k)
            new_lam_bundles(step, k, 0) = initial_data(step, 8 + k);
        time[step] = initial_data(step, 0);
    }

    // Kokkos view for sigma points
    Kokkos::View<double****, MEMORY_SPACE> sigmas_combined("sigmas_combined", num_bundles, num_sigma, nsd, num_steps);

    // Setup time steps
    Kokkos::View<int*, MEMORY_SPACE> time_steps_view("time_steps_view", num_steps);  // Kokkos::View<int*> as expected
    auto time_steps_host = Kokkos::create_mirror_view(time_steps_view);
    for (int i = 0; i < num_steps; ++i)
        time_steps_host(i) = i;
    Kokkos::deep_copy(time_steps_view, time_steps_host);

    // Setup time view (Kokkos::View<double*> to hold the time values)
    Kokkos::View<double*, MEMORY_SPACE> time_view("time_view", num_steps);  // Kokkos::View<double*>
    for (int step = 0; step < num_steps; ++step) {
        time_view(step) = initial_data(step, 0);  // Time values in the `time_view`
    }

    // Covariance matrices
    Eigen::MatrixXd P_pos = 0.01 * Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd P_vel = 0.0001 * Eigen::MatrixXd::Identity(3, 3);
    double P_mass = 0.0001;

    double P_pos_flat[9], P_vel_flat[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            P_pos_flat[i*3 + j] = P_pos(i, j);
            P_vel_flat[i*3 + j] = P_vel(i, j);
        }


    generate_sigma_points_kokkos(
        nsd, alpha, beta, kappa,
        P_pos_flat, P_vel_flat, P_mass,
        time_steps_view, r_bundles, v_bundles, m_bundles,
        sigmas_combined
    );

    // Propagation settings
    PropagationSettings settings;
    settings.mu = 27.899633640439433;
    settings.F = 0.33;
    settings.c = 4.4246246663455135;
    settings.m0 = 4000.0;
    settings.g0 = 9.81;
    settings.num_subintervals = 10;
    settings.num_eval_per_step = 200;
    Kokkos::View<double****, MEMORY_SPACE> trajectories_out("trajectories_out", num_bundles, num_sigma, settings.num_eval_per_step, 8);

    // Sample random control inputs
    const int num_random_samples = (num_steps - 1) * (settings.num_subintervals - 1);
    auto random_controls_host = Kokkos::View<double**, Kokkos::HostSpace>("random_controls_host", num_random_samples, 7);
    sample_controls_host_host(num_random_samples, random_controls_host);

    Kokkos::View<double**, MEMORY_SPACE> random_controls("random_controls", num_random_samples, 7);
    Kokkos::deep_copy(random_controls, random_controls_host);

    // Create transform matrix
    Kokkos::View<double**, MEMORY_SPACE> transform("transform", 7, 7);
    compute_transform_matrix(transform);

    // Create views for Wm and Wc
    Kokkos::View<double*, MEMORY_SPACE> Wm_view("Wm", num_sigma);
    Kokkos::View<double*, MEMORY_SPACE> Wc_view("Wc", num_sigma);
    auto Wm_host = Kokkos::create_mirror_view(Wm_view);
    auto Wc_host = Kokkos::create_mirror_view(Wc_view);
    for (int i = 0; i < num_sigma; ++i) {
        Wm_host(i) = Wm[i];
        Wc_host(i) = Wc[i];
    }
    Kokkos::deep_copy(Wm_view, Wm_host);
    Kokkos::deep_copy(Wc_view, Wc_host);

    // Propagate sigma point trajectories
    propagate_sigma_trajectories(
        sigmas_combined, new_lam_bundles,
        time_view, Wm_view, Wc_view,
        random_controls, transform, settings,
        trajectories_out
    );

    // Copy output back to host for verification
    auto host_traj = Kokkos::create_mirror_view(trajectories_out);
    Kokkos::deep_copy(host_traj, trajectories_out);

    // Load expected data for comparison
    auto expected_data_vec = load_csv("expected_trajectories_bundle_32.csv", 10); 
    Eigen::MatrixXd expected_data(expected_data_vec.size(), expected_data_vec[0].size());
    for (int i = 0; i < expected_data.rows(); ++i)
        for (int j = 0; j < expected_data.cols(); ++j)
            expected_data(i, j) = expected_data_vec[i][j];

    // Check the results
    double tol = 1e-1;
    for (int step = 0; step < settings.num_eval_per_step; ++step) {
        for (int d = 0; d < 7; ++d) {
            double actual = host_traj(0, 0, step, d);
            double expected = expected_data(step, d + 2);
            CHECK_THAT(actual, Catch::Matchers::WithinAbs(expected, tol));
        }
        double actual_time = host_traj(0, 0, step, 7);
        double expected_time = expected_data(step, 9);
        CHECK_THAT(actual_time, Catch::Matchers::WithinAbs(expected_time, tol));
    }

    SUCCEED("Checked propagated values for Trajectory 32.");
}
