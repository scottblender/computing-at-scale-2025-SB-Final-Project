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
#include "csv_loader.hpp"

TEST_CASE("Sigma point propagation first row check (with weights)", "[propagation]") {
    Eigen::MatrixXd initial_data = load_csv_matrix("initial_bundle_32.csv");
    Eigen::MatrixXd expected = load_csv_matrix("expected_trajectories_full.csv");

    std::vector<double> Wm, Wc;
    load_weights("sigma_weights.csv", Wm, Wc);

    const int test_bundle = 32;
    const int num_bundles = 1;
    const int num_sigma = static_cast<int>(Wm.size());
    const int num_steps = initial_data.rows() / num_sigma;
    const int num_storage_steps = expected.rows() / num_sigma;
    const int evals_per_step = num_storage_steps / (num_steps - 1);

    std::vector<double> time;
    for (int i = 0; i < num_steps; ++i)
        time.push_back(initial_data(i * num_sigma, 0));

    Kokkos::View<double****> sigmas_combined("sigmas_combined", num_bundles, num_sigma, 7, num_steps);
    Kokkos::View<double***> new_lam_bundles("new_lam_bundles", num_steps, 7, num_bundles);
    Kokkos::View<double****> trajectories_out("trajectories_out", num_bundles, num_sigma, num_storage_steps, 8);

    for (int sigma = 0; sigma < num_sigma; ++sigma) {
        for (int step = 0; step < num_steps; ++step) {
            int row = step * num_sigma + sigma;
            for (int k = 0; k < 7; ++k)
                sigmas_combined(0, sigma, k, step) = initial_data(row, k + 1);
            for (int k = 0; k < 7; ++k)
                new_lam_bundles(step, k, 0) = initial_data(row, k + 8);
        }
    }

    PropagationSettings settings;
    settings.mu = 27.899633640439433;
    settings.F = 0.33;
    settings.c = 4.4246246663455135;
    settings.m0 = 4000.0;
    settings.g0 = 9.81;
    settings.num_eval_per_step = evals_per_step;
    settings.state_size = 7;
    settings.control_size = 7;

    propagate_sigma_trajectories(sigmas_combined, new_lam_bundles, time, Wm, Wc, settings, trajectories_out);

    auto host_traj = Kokkos::create_mirror_view(trajectories_out);
    Kokkos::deep_copy(host_traj, trajectories_out);

    // Only check first row
    int row = 0;
    int bundle = static_cast<int>(expected(row, 0));
    if (bundle == test_bundle) {
        int sigma = static_cast<int>(expected(row, 1));
        double t_val = expected(row, expected.cols() - 1);
        double t_start = time.front();
        double t_end = time.back();
        double step_size = (t_end - t_start) / (num_storage_steps - 1);
        int step = static_cast<int>((t_val - t_start) / step_size + 0.5);

        REQUIRE(step >= 0);
        REQUIRE(step < num_storage_steps);

        for (int d = 0; d < 8; ++d) {
            double actual = host_traj(0, sigma, step, d);
            double reference = expected(row, d + 2);
            INFO("Mismatch at bundle " << bundle << ", sigma " << sigma
                 << ", step " << step << ", dim " << d
                 << ", actual=" << actual << ", expected=" << reference);
            CHECK_THAT(actual, Catch::Matchers::WithinAbs(reference, 1e-3));
        }
    } else {
        FAIL("First row of expected trajectory CSV does not match test bundle.");
    }
}
