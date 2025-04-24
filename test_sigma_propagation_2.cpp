#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "csv_loader.hpp"
#include "sigma_propagation.hpp"

TEST_CASE("Check propagated result at first expected row index", "[propagation]") {
    // Load expected CSV (for reference row)
    Eigen::MatrixXd expected = load_csv_matrix("expected_trajectories_full.csv");

    // Load weights
    std::vector<double> Wm, Wc;
    load_weights("sigma_weights.csv", Wm, Wc);

    // Load initial data
    Eigen::MatrixXd initial_data = load_csv_matrix("initial_bundle_32.csv");

    // Setup parameters
    const int num_bundles = 1;
    const int num_sigma = static_cast<int>(Wm.size());
    const int num_steps = initial_data.rows() / num_sigma;
    const int num_storage_steps = expected.rows() / num_sigma;
    const int evals_per_step = num_storage_steps / (num_steps - 1);

    // Time vector
    std::vector<double> time;
    for (int i = 0; i < num_steps; ++i)
        time.push_back(initial_data(i * num_sigma, 0));

    // Allocate views
    Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, 7, num_steps);
    Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);
    Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, num_storage_steps, 8);

    // Populate sigmas and lambdas
    for (int sigma = 0; sigma < num_sigma; ++sigma) {
        for (int step = 0; step < num_steps; ++step) {
            int row = step * num_sigma + sigma;
            for (int k = 0; k < 7; ++k)
                sigmas_combined(0, sigma, k, step) = initial_data(row, k + 1);
            for (int k = 0; k < 7; ++k)
                new_lam_bundles(step, k, 0) = initial_data(row, k + 8);
        }
    }

    // Define propagation settings
    PropagationSettings settings;
    settings.mu = 27.899633640439433;
    settings.F = 0.33;
    settings.c = 4.4246246663455135;
    settings.m0 = 4000.0;
    settings.g0 = 9.81;
    settings.num_eval_per_step = evals_per_step;
    settings.state_size = 7;
    settings.control_size = 7;

    // Perform propagation
    propagate_sigma_trajectories(sigmas_combined, new_lam_bundles, time, Wm, Wc, settings, trajectories_out);

    auto host_traj = Kokkos::create_mirror_view(trajectories_out);
    Kokkos::deep_copy(host_traj, trajectories_out);

    // --- Extract target (bundle, sigma, time) from expected ---
    int bundle = static_cast<int>(expected(0, 0));
    int sigma = static_cast<int>(expected(0, 1));
    double t_val = expected(0, expected.cols() - 1);

    double t_start = time.front();
    double t_end = time.back();
    double step_size = (t_end - t_start) / (num_storage_steps - 1);
    int step = static_cast<int>((t_val - t_start) / step_size + 0.5);

    REQUIRE(bundle == 32);  // Confirming this is the bundle used
    REQUIRE(step >= 0);
    REQUIRE(step < num_storage_steps);

    // --- Print the propagated result ---
    std::cout << "\nPropagated results at bundle=" << bundle << ", sigma=" << sigma << ", step=" << step << ":\n";
    for (int d = 0; d < 8; ++d) {
        std::cout << "  Dim[" << d << "] = " << host_traj(0, sigma, step, d) << '\n';
    }

    SUCCEED("Successfully accessed propagated values for comparison.");
}
