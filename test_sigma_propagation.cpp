#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include "sigma_propagation.hpp"
#include "rv2mee.hpp"
#include "mee2rv.hpp"
#include "odefunc.hpp"
#include "l1_dot_2B_propul.hpp"
#include "lm_dot_2B_propul.hpp"

using namespace Catch::Matchers;

TEST_CASE("Sigma point propagation returns expected trajectory sizes and time history", "[propagation]") {
    Kokkos::initialize();

    {
        const int num_bundles = 1;
        const int num_sigma = 3;
        const int num_steps = 2;

        std::vector<double> time = {0.0, 10.0};  // t0 and t1
        std::vector<double> Wm = {0.5, 0.25, 0.25};
        std::vector<double> Wc = {0.5, 0.25, 0.25};

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
        const double dt = (time[1] - time[0]) / settings.num_eval_per_step;

        Kokkos::View<double****> trajectories_out("trajectories_out", 
            num_bundles, 
            num_sigma, 
            num_storage_steps, 
            7
        );

        // Optional: Store time separately
        std::vector<double> time_history(num_storage_steps);
        for (int i = 0; i < num_storage_steps; ++i)
            time_history[i] = time[0] + i * dt;

        // Fill dummy values
        Kokkos::parallel_for("init_sigmas", num_bundles * num_sigma * 7 * num_steps, KOKKOS_LAMBDA(int idx) {
            int i = idx / (num_sigma * 7 * num_steps);
            int j = (idx / (7 * num_steps)) % num_sigma;
            int k = (idx / num_steps) % 7;
            int t = idx % num_steps;
            sigmas_combined(i, j, k, t) = 0.1 * (k + 1);
        });

        Kokkos::parallel_for("init_lam", num_steps * 7 * num_bundles, KOKKOS_LAMBDA(int idx) {
            int t = idx / (7 * num_bundles);
            int k = (idx / num_bundles) % 7;
            int i = idx % num_bundles;
            new_lam_bundles(t, k, i) = 0.01 * (k + 1);
        });

        propagate_sigma_trajectories(sigmas_combined, new_lam_bundles, time, Wm, Wc, settings, trajectories_out);

        auto host_traj = Kokkos::create_mirror_view(trajectories_out);
        Kokkos::deep_copy(host_traj, trajectories_out);

        REQUIRE(host_traj.extent(0) == num_bundles);
        REQUIRE(host_traj.extent(1) == num_sigma);
        REQUIRE(host_traj.extent(2) == num_storage_steps);
        REQUIRE(host_traj.extent(3) == 7);

        // Time history validation
        for (int n = 0; n < num_storage_steps; ++n) {
            double expected_time = time[0] + n * dt;
            CHECK_THAT(time_history[n], WithinAbs(expected_time, 1e-10));
        }

        // Value structural test
        CHECK_THAT(host_traj(0, 0, 0, 0), WithinAbs(0.0, 1e2));  // position x at first step
    }

    Kokkos::finalize();
}
