#include <catch2/catch_test_macros.hpp>
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

TEST_CASE("Print propagated values for bundle=32, sigma=0 for single interval [GPU-compatible]", "[propagation]") {
    Eigen::MatrixXd initial_data = load_csv("initial_bundle_32.csv", 16);

    std::vector<double> Wm, Wc;
    load_weights("sigma_weights.csv", Wm, Wc);

    const int num_sigma = static_cast<int>(Wm.size());
    const int num_steps = 2;
    const int num_bundles = 1;
    const int nsd = 7;

    Kokkos::View<double***> r_bundles("r_bundles", num_bundles, num_steps, 3);
    Kokkos::View<double***> v_bundles("v_bundles", num_bundles, num_steps, 3);
    Kokkos::View<double**> m_bundles("m_bundles", num_bundles, num_steps);
    Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);
    std::vector<double> time(num_steps);

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

    Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, 7, num_steps);
    std::vector<int> time_steps = {0, 1};

    double alpha = 1.7215, beta = 2.0, kappa = 3.0 - nsd;
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
        time_steps, r_bundles, v_bundles, m_bundles,
        sigmas_combined
    );
    double P_mass = 0.0001;

    generate_sigma_points_kokkos(
        nsd, alpha, beta, kappa,
        P_pos, P_vel, P_mass,
        time_steps, r_bundles, v_bundles, m_bundles,
        sigmas_combined
    );

    PropagationSettings settings;
    settings.mu = 27.899633640439433;
    settings.F = 0.33;
    settings.c = 4.4246246663455135;
    settings.m0 = 4000.0;
    settings.g0 = 9.81;
    settings.num_subintervals = 10;
    settings.num_eval_per_step = 200;

    int num_storage_steps = settings.num_eval_per_step;
    Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, num_storage_steps, 8);

    // Prepare random controls and transform
    const int num_random_samples = (num_steps - 1) * (settings.num_subintervals - 1);
    Kokkos::View<double**> random_controls("random_controls", num_random_samples, 7);
    Kokkos::View<double**> transform("transform", 7, 7);
    sample_controls_host(num_random_samples, random_controls);
    compute_transform_matrix(transform);

    // Prepare time and weights views
    Kokkos::View<double*> time_view("time_view", num_steps);
    auto time_host = Kokkos::create_mirror_view(time_view);
    for (int i = 0; i < num_steps; ++i)
        time_host(i) = time[i];
    Kokkos::deep_copy(time_view, time_host);

    Kokkos::View<double*> Wm_view("Wm", num_sigma);
    Kokkos::View<double*> Wc_view("Wc", num_sigma);
    auto Wm_host = Kokkos::create_mirror_view(Wm_view);
    auto Wc_host = Kokkos::create_mirror_view(Wc_view);
    for (int i = 0; i < num_sigma; ++i) {
        Wm_host(i) = Wm[i];
        Wc_host(i) = Wc[i];
    }
    Kokkos::deep_copy(Wm_view, Wm_host);
    Kokkos::deep_copy(Wc_view, Wc_host);

    // Propagate
    propagate_sigma_trajectories(
        sigmas_combined, new_lam_bundles,
        time_view, Wm_view, Wc_view,
        random_controls, transform, settings,
        trajectories_out
    );

    // Copy output back to host
    auto host_traj = Kokkos::create_mirror_view(trajectories_out);
    Kokkos::deep_copy(host_traj, trajectories_out);

    // === Print the results ===
    int sigma_to_print = 0;
    for (int step = 0; step < num_storage_steps; ++step) {
        std::cout << "\nStep " << step
                  << " at time = " << host_traj(0, sigma_to_print, step, 7) << '\n';
        for (int d = 0; d < 7; ++d) {
            double value = host_traj(0, sigma_to_print, step, d);
            std::cout << "  Dim[" << d << "] = " << value << '\n';
        }
    }

    SUCCEED("Printed propagated values for Trajectory 32.");
}
