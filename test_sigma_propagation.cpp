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

// Utility to load CSV into Eigen
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

TEST_CASE("Sigma point propagation matches reference integration", "[propagation]") {
    Kokkos::initialize();

    {
        const int num_bundles = 1;
        const int num_sigma = 1;
        const int num_steps = 2;

        std::vector<double> time = {0.0, 10.0};
        std::vector<double> Wm = {1.0};
        std::vector<double> Wc = {1.0};

        Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, 7, num_steps);
        Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);

        PropagationSettings settings;
        settings.mu = 398600.4418;
        settings.F = 1.0;
        settings.c = 300.0;
        settings.m0 = 1000.0;
        settings.g0 = 9.80665;
        settings.num_eval_per_step = 20;
        settings.state_size = 7;
        settings.control_size = 7;

        const int num_storage_steps = (num_steps - 1) * (settings.num_eval_per_step + 1);

        Kokkos::View<double****> trajectories_out("trajectories_out", 
            num_bundles, 
            num_sigma, 
            num_storage_steps, 
            8
        );

        // Initialize with same values as reference
        Kokkos::parallel_for("init_sigmas", 7, KOKKOS_LAMBDA(int k) {
            sigmas_combined(0, 0, k, 0) = 0.1 * (k + 1);
            sigmas_combined(0, 0, k, 1) = 0.1 * (k + 1);
        });

        Kokkos::parallel_for("init_lam", 7 * num_steps, KOKKOS_LAMBDA(int idx) {
            int t = idx / 7;
            int k = idx % 7;
            new_lam_bundles(t, k, 0) = 0.01 * (k + 1);
        });

        propagate_sigma_trajectories(sigmas_combined, new_lam_bundles, time, Wm, Wc, settings, trajectories_out);

        auto host_traj = Kokkos::create_mirror_view(trajectories_out);
        Kokkos::deep_copy(host_traj, trajectories_out);

        REQUIRE(host_traj.extent(0) == num_bundles);
        REQUIRE(host_traj.extent(1) == num_sigma);
        REQUIRE(host_traj.extent(2) == num_storage_steps);
        REQUIRE(host_traj.extent(3) == 8);

        // Load expected CSV
        Eigen::MatrixXd expected = load_csv("expected_trajectory.csv", num_storage_steps, 8);

        // Compare each timestep
        for (int n = 0; n < num_storage_steps; ++n) {
            for (int d = 0; d < 8; ++d) {
                double actual = host_traj(0, 0, n, d);
                double ref = expected(n, d);
                INFO("Mismatch at step " << n << ", dim " << d);
                CHECK_THAT(actual, Catch::Matchers::WithinAbs(ref, 1e-6));
            }
        }
    }

    Kokkos::finalize();
}