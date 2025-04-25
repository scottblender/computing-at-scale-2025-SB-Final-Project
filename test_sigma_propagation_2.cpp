#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#include "csv_loader.hpp"
#include "sigma_propagation.hpp"
#include "sigma_points_kokkos.hpp"

TEST_CASE("Print propagated values for bundle=32, sigma=0 for single interval", "[propagation]") {
    Eigen::MatrixXd expected = load_csv_matrix("expected_trajectories_full.csv");
    Eigen::MatrixXd initial_data = load_csv_matrix("initial_bundle_32.csv");

    std::vector<double> Wm, Wc;
    load_weights("sigma_weights.csv", Wm, Wc);

    const int num_sigma = static_cast<int>(Wm.size());
    const int num_steps = 2;  // Only propagate from time[0] -> time[1]
    const int num_bundles = 1;
    const int nsd = 7;

    Kokkos::View<double***> r_bundles("r_bundles", num_bundles, num_steps, 3);
    Kokkos::View<double***> v_bundles("v_bundles", num_bundles, num_steps, 3);
    Kokkos::View<double**> m_bundles("m_bundles", num_bundles, num_steps);
    Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);
    std::vector<double> time(num_steps);

    // Fill forward in time
    for (int step = 0; step < num_steps; ++step) {
        int row = step * num_sigma;  // sigma=0 row for step
        for (int k = 0; k < 3; ++k) {
            r_bundles(0, step, k) = initial_data(row, 1 + k);
            v_bundles(0, step, k) = initial_data(row, 4 + k);
        }
        m_bundles(0, step) = initial_data(row, 7);
        for (int k = 0; k < 7; ++k)
            new_lam_bundles(step, k, 0) = initial_data(row, 8 + k);
        time[step] = initial_data(row, 0);
    }

    // Sigma point generation
    Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, 7, num_steps);
    std::vector<int> time_steps = {0, 1};

    double alpha = 1.7215, beta = 2.0, kappa = 3.0 - nsd;
    Eigen::MatrixXd P_pos = 0.01 * Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd P_vel = 0.0001 * Eigen::MatrixXd::Identity(3, 3);
    double P_mass = 0.0001;

    generate_sigma_points_kokkos(
        nsd, alpha, beta, kappa,
        P_pos, P_vel, P_mass,
        time_steps, r_bundles, v_bundles, m_bundles,
        sigmas_combined
    );

    // Propagation
    PropagationSettings settings;
    settings.mu = 27.899633640439433;
    settings.F = 0.33;
    settings.c = 4.4246246663455135;
    settings.m0 = 4000.0;
    settings.g0 = 9.81;
    settings.num_eval_per_step = 200;
    settings.state_size = 7;
    settings.control_size = 7;

    // Propagation will output exactly num_eval_per_step + 1 time points
    int num_storage_steps = settings.num_eval_per_step + 1;
    Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, num_storage_steps, 8);

    propagate_sigma_trajectories(sigmas_combined, new_lam_bundles, time, Wm, Wc, settings, trajectories_out);

    auto host_traj = Kokkos::create_mirror_view(trajectories_out);
    Kokkos::deep_copy(host_traj, trajectories_out);

    int expected_sigma = static_cast<int>(expected(0, 1));

    // Optional validation of time range
    REQUIRE(host_traj(0, expected_sigma, 0, 7) == Approx(time[0]).margin(1e-10));
    REQUIRE(host_traj(0, expected_sigma, num_storage_steps - 1, 7) == Approx(time[1]).margin(1e-10));

    // Print results
    for (int step = 0; step < num_storage_steps; ++step) {
        std::cout << "\nStep " << step << " at time = " << host_traj(0, expected_sigma, step, 7) << '\n';
        for (int d = 0; d < 8; ++d)
            std::cout << "  Dim[" << d << "] = " << host_traj(0, expected_sigma, step, d) << '\n';
    }

    SUCCEED("Single-step propagation executed and printed.");
}
