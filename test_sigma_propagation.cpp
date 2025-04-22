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

// Load CSV utility
Eigen::MatrixXd load_csv(const std::string& path, int rows, int cols) {
    Eigen::MatrixXd mat(rows, cols);
    std::ifstream file(path);
    std::string line;
    int row = 0;

    while (std::getline(file, line) && row < rows) {
        std::stringstream ss(line);
        std::string value;
        int col = 0;
        while (std::getline(ss, value, ',') && col < cols) {
            mat(row, col) = std::stod(value);
            col++;
        }
        row++;
    }
    return mat;
}

TEST_CASE("Sigma point propagation matches expected trajectory output", "[propagation]") {
    Kokkos::initialize();

    {
        const int num_bundles = 1;
        const int num_sigma = 1;
        const int num_steps = 2;
        const int evals_per_step = 2;

        std::vector<double> time = {0.0, 2.0};
        std::vector<double> Wm = {1.0};
        std::vector<double> Wc = {1.0};

        PropagationSettings settings;
        settings.mu = 398600.4418;
        settings.F = 0.0;  // disable thrust to match dummy CSV
        settings.c = 300.0;
        settings.m0 = 1000.0;
        settings.g0 = 9.80665;
        settings.num_eval_per_step = evals_per_step;
        settings.state_size = 7;
        settings.control_size = 7;

        const int num_storage_steps = (num_steps - 1) * (evals_per_step + 1);

        // [bundle][sigma][step][x, y, z, vx, vy, vz, m, t]
        Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, 7, num_steps);
        Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);
        Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, num_storage_steps, 8);

        // Initialize initial state: r = [1000, 2000, 3000], v = [1, 2, 3], m = 500
        Kokkos::parallel_for("init_sigmas", 7, KOKKOS_LAMBDA(int k) {
            sigmas_combined(0, 0, k, 0) = (k < 6) ? ((k < 3) ? 1000.0 + k * 1000.0 : 1.0 + (k - 3)) : 500.0;
            sigmas_combined(0, 0, k, 1) = sigmas_combined(0, 0, k, 0);  // redundant but consistent
        });

        Kokkos::parallel_for("init_lam", 14, KOKKOS_LAMBDA(int idx) {
            int t = idx / 7;
            int k = idx % 7;
            new_lam_bundles(t, k, 0) = 0.0;  // no control input
        });

        propagate_sigma_trajectories(sigmas_combined, new_lam_bundles, time, Wm, Wc, settings, trajectories_out);

        auto host_traj = Kokkos::create_mirror_view(trajectories_out);
        Kokkos::deep_copy(host_traj, trajectories_out);

        REQUIRE(host_traj.extent(0) == num_bundles);
        REQUIRE(host_traj.extent(1) == num_sigma);
        REQUIRE(host_traj.extent(2) == num_storage_steps);
        REQUIRE(host_traj.extent(3) == 8);  // x, y, z, vx, vy, vz, m, t

        Eigen::MatrixXd expected = load_csv("expected_trajectory.csv", num_storage_steps, 9);

        for (int step = 0; step < num_storage_steps; ++step) {
            for (int d = 0; d < 8; ++d) {
                double actual = host_traj(0, 0, step, d);
                double reference = expected(step, d);
                INFO("Mismatch at step " << step << ", dim " << d);
                CHECK_THAT(actual, Catch::Matchers::WithinAbs(reference, 1e-6));
            }
        }
    }

    Kokkos::finalize();
}
