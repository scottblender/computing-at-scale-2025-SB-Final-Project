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

Eigen::MatrixXd load_csv(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    std::vector<std::vector<double>> rows;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        rows.push_back(row);
    }

    Eigen::MatrixXd mat(rows.size(), rows[0].size());
    for (int i = 0; i < rows.size(); ++i)
        for (int j = 0; j < rows[i].size(); ++j)
            mat(i, j) = rows[i][j];

    return mat;
}

TEST_CASE("Sigma point propagation matches expected trajectory output", "[propagation]") {
    Kokkos::initialize();

    {
        const int num_bundles = 2;
        const int num_sigma = 1;
        const int num_steps = 2;
        const int evals_per_step = 2;

        std::vector<double> time = {0.0, 2.0};
        std::vector<double> Wm = {1.0};
        std::vector<double> Wc = {1.0};

        PropagationSettings settings;
        settings.mu = 398600.4418;
        settings.F = 0.0;  // no thrust to match expected output
        settings.c = 300.0;
        settings.m0 = 1000.0;
        settings.g0 = 9.80665;
        settings.num_eval_per_step = evals_per_step;
        settings.state_size = 7;
        settings.control_size = 7;

        const int num_storage_steps = (num_steps - 1) * (evals_per_step + 1);

        Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, 7, num_steps);
        Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);
        Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, num_storage_steps, 8);

        // Bundle 0: initial state 1
        Kokkos::parallel_for("init_sigma_0", 7, KOKKOS_LAMBDA(int k) {
            sigmas_combined(0, 0, k, 0) = (k < 6) ? ((k < 3) ? 1000.0 + k * 1000.0 : 1.0 + (k - 3)) : 500.0;
            sigmas_combined(0, 0, k, 1) = sigmas_combined(0, 0, k, 0);
        });

        // Bundle 1: initial state 2
        Kokkos::parallel_for("init_sigma_1", 7, KOKKOS_LAMBDA(int k) {
            sigmas_combined(1, 0, k, 0) = (k < 6) ? ((k < 3) ? 2000.0 + k * 1000.0 : 1.5 + (k - 3)) : 600.0;
            sigmas_combined(1, 0, k, 1) = sigmas_combined(1, 0, k, 0);
        });

        Kokkos::parallel_for("init_lam", num_steps * 7 * num_bundles, KOKKOS_LAMBDA(int idx) {
            int t = idx / (7 * num_bundles);
            int k = (idx / num_bundles) % 7;
            int b = idx % num_bundles;
            new_lam_bundles(t, k, b) = 0.0;
        });

        propagate_sigma_trajectories(sigmas_combined, new_lam_bundles, time, Wm, Wc, settings, trajectories_out);

        auto host_traj = Kokkos::create_mirror_view(trajectories_out);
        Kokkos::deep_copy(host_traj, trajectories_out);

        REQUIRE(host_traj.extent(0) == num_bundles);
        REQUIRE(host_traj.extent(1) == num_sigma);
        REQUIRE(host_traj.extent(2) == num_storage_steps);
        REQUIRE(host_traj.extent(3) == 8);

        Eigen::MatrixXd expected = load_csv("expected_trajectory_full.csv");
        REQUIRE(expected.cols() == 10);  // bundle, sigma, x, y, z, vx, vy, vz, m, t

        for (int row = 0; row < expected.rows(); ++row) {
            int bundle = static_cast<int>(expected(row, 0));
            int sigma = static_cast<int>(expected(row, 1));
            int step = row % num_storage_steps;
            for (int d = 0; d < 8; ++d) {
                double actual = host_traj(bundle, sigma, step, d);
                double reference = expected(row, d + 2);
                INFO("Mismatch at bundle " << bundle << ", sigma " << sigma << ", step " << step << ", dim " << d);
                CHECK_THAT(actual, Catch::Matchers::WithinAbs(reference, 1e-6));
            }
        }
    }

    Kokkos::finalize();
}
