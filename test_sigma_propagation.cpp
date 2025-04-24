#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "sigma_propagation.hpp"
#include "rv2mee.hpp"
#include "mee2rv.hpp"
#include "odefunc.hpp"
#include "l1_dot_2B_propul.hpp"
#include "lm_dot_2B_propul.hpp"
#include "csv_loader.hpp"  // contains load_weights()

// CSV loader for Eigen matrix
Eigen::MatrixXd load_csv_matrix(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    std::vector<std::vector<double>> rows;

    // Skip header
    std::getline(file, line);

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

TEST_CASE("Sigma point propagation matches expected CSV output (layout inferred)", "[propagation]") {
    Eigen::MatrixXd initial_data = load_csv_matrix("initial_bundle_32.csv");
    Eigen::MatrixXd expected = load_csv_matrix("expected_trajectories_full.csv");

    std::vector<double> Wm, Wc;
    load_weights("sigma_weights.csv", Wm, Wc);

    const int num_bundles = 1;
    const int num_sigma = static_cast<int>(Wm.size());
    const int num_steps = initial_data.rows() / num_sigma;
    const int num_storage_steps = expected.rows() / (num_bundles * num_sigma);
    const int evals_per_step = num_storage_steps / (num_steps - 1);

    std::vector<double> time;
    for (int i = 0; i < num_steps; ++i)
        time.push_back(initial_data(i * num_sigma, 0));  // one time value per step

    // Allocate views
    Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, 7, num_steps);
    Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);
    Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, num_storage_steps, 8);

    // Fill in sigmas and controls
    for (int sigma = 0; sigma < num_sigma; ++sigma) {
        for (int step = 0; step < num_steps; ++step) {
            int row = step * num_sigma + sigma;
            for (int k = 0; k < 7; ++k)
                sigmas_combined(0, sigma, k, step) = initial_data(row, k + 1);
            for (int k = 0; k < 7; ++k)
                new_lam_bundles(step, k, 0) = initial_data(row, k + 8);
        }
    }

    // Settings
    PropagationSettings settings;
    settings.mu = 27.899633640439433;
    settings.F = 0.0; // For test, disable thrust
    settings.c = 4.4246246663455135;
    settings.m0 = 4000.0;
    settings.g0 = 9.81;
    settings.num_eval_per_step = evals_per_step;
    settings.state_size = 7;
    settings.control_size = 7;

    // Run propagation
    propagate_sigma_trajectories(sigmas_combined, new_lam_bundles, time, Wm, Wc, settings, trajectories_out);

    auto host_traj = Kokkos::create_mirror_view(trajectories_out);
    Kokkos::deep_copy(host_traj, trajectories_out);

    // Verify dimensions
    REQUIRE(host_traj.extent(0) == num_bundles);
    REQUIRE(host_traj.extent(1) == num_sigma);
    REQUIRE(host_traj.extent(2) == num_storage_steps);
    REQUIRE(host_traj.extent(3) == 8);

    // Check against expected CSV
    for (int row = 0; row < expected.rows(); ++row) {
        int bundle = static_cast<int>(expected(row, 0));
        int sigma  = static_cast<int>(expected(row, 1));
        double time_val = expected(row, 17); // last column is time
        int step = -1;

        // Find closest time index
        for (int i = 0; i < num_storage_steps; ++i) {
            double t_curr = host_traj(bundle, sigma, i, 7); // stored time
            if (std::abs(t_curr - time_val) < 1e-8) {
                step = i;
                break;
            }
        }

        REQUIRE(step != -1); // ensure we matched a step

        for (int d = 0; d < 7; ++d) {
            double actual = host_traj(bundle, sigma, step, d);
            double reference = expected(row, d + 2); // skip bundle,sigma
            INFO("Mismatch at bundle " << bundle << ", sigma " << sigma << ", step " << step << ", dim " << d);
            CHECK_THAT(actual, Catch::Matchers::WithinAbs(reference, 1e-6));
        }

        // Optional: check time too
        double actual_time = host_traj(bundle, sigma, step, 7);
        CHECK_THAT(actual_time, Catch::Matchers::WithinAbs(time_val, 1e-8));
    }
}
