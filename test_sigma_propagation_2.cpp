#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include "sigma_propagation.hpp"
#include "rv2mee.hpp"
#include "mee2rv.hpp"
#include "odefunc.hpp"
#include "l1_dot_2B_propul.hpp"
#include "lm_dot_2B_propul.hpp"

using Catch::Matchers::WithinAbs;

Eigen::MatrixXd load_csv(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    std::vector<std::vector<double>> rows;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        while (std::getline(ss, value, ','))
            row.push_back(std::stod(value));
        rows.push_back(row);
    }
    Eigen::MatrixXd mat(rows.size(), rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j)
            mat(i, j) = rows[i][j];
    return mat;
}

TEST_CASE("Compare propagated sigma trajectories to expected CSV output", "[propagation]") {
    Kokkos::initialize();
    {
        // Load all relevant CSV files
        Eigen::MatrixXd initial = load_csv("initial_sigma_states_bundle32.csv");
        Eigen::MatrixXd expected = load_csv("expected_trajectory_full_bundle32.csv");
        Eigen::MatrixXd controls = load_csv("new_lam_bundles_bundle32.csv");
        Eigen::MatrixXd weights = load_csv("sigma_weights.csv");

        int num_sigma = weights.rows();
        int num_steps = expected.rows() / num_sigma;
        int num_bundles = 1;
        int control_dim = 7;
        int state_dim = 7;
        int evals_per_step = num_steps - 1;

        // Set up time vector
        std::vector<double> time(num_steps);
        for (int i = 0; i < num_steps; ++i)
            time[i] = expected(i, 9); // last column is time

        // Extract weights
        std::vector<double> Wm(num_sigma), Wc(num_sigma);
        for (int i = 0; i < num_sigma; ++i) {
            Wm[i] = weights(i, 1);
            Wc[i] = weights(i, 2);
        }

        // Initialize sigmas_combined and new_lam_bundles
        Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, 7, num_steps);
        Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);
        Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, num_steps, 8);

        // Fill sigmas_combined from initial
        for (int sigma = 0; sigma < num_sigma; ++sigma) {
            for (int i = 0; i < 7; ++i)
                sigmas_combined(0, sigma, i, 0) = initial(sigma, i + 1); // skip time
        }

        // Copy control inputs
        for (int t = 0; t < num_steps; ++t)
            for (int k = 0; k < 7; ++k)
                new_lam_bundles(t, k, 0) = controls(t, k + 2);

        // Set propagation settings
        PropagationSettings settings;
        settings.mu = 398600.4418;
        settings.F = 1.0;
        settings.c = 300.0;
        settings.m0 = 1000.0;
        settings.g0 = 9.80665;
        settings.num_eval_per_step = evals_per_step;
        settings.state_size = 7;
        settings.control_size = 7;

        // Run propagation
        propagate_sigma_trajectories(sigmas_combined, new_lam_bundles, time, Wm, Wc, settings, trajectories_out);

        auto host_traj = Kokkos::create_mirror_view(trajectories_out);
        Kokkos::deep_copy(host_traj, trajectories_out);

        // Verify trajectory shape
        REQUIRE(host_traj.extent(0) == 1);
        REQUIRE(host_traj.extent(1) == num_sigma);
        REQUIRE(host_traj.extent(2) == num_steps);
        REQUIRE(host_traj.extent(3) == 8);

        // Compare each propagated entry to expected
        for (int row = 0; row < expected.rows(); ++row) {
            int bundle = static_cast<int>(expected(row, 0));
            int sigma = static_cast<int>(expected(row, 1));
            for (int d = 0; d < 8; ++d) {
                double actual = host_traj(bundle, sigma, row % num_steps, d);
                double reference = expected(row, d + 2); // skip bundle, sigma
                INFO("Mismatch at row " << row << ", dim " << d);
                CHECK_THAT(actual, WithinAbs(reference, 1e-6));
            }
        }
    }
    Kokkos::finalize();
}
