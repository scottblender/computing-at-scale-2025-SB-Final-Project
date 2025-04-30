#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "../include/csv_loader.hpp"
#include "../include/sigma_points_kokkos.hpp"

using namespace Catch::Matchers;

TEST_CASE("sigma_points_kokkos generates expected number of sigma points [bundle 32]", "[sigma]") {
    // === Load CSV ===
    Eigen::MatrixXd full_data = load_csv_matrix("initial_bundles_all.csv");

    // === Filter bundle 32 rows ===
    std::vector<Eigen::VectorXd> bundle_32_rows;
    for (int i = 0; i < full_data.rows(); ++i) {
        if (static_cast<int>(full_data(i, 15)) == 32)
            bundle_32_rows.push_back(full_data.row(i));
    }

    REQUIRE(bundle_32_rows.size() >= 3);  // must have at least 3 time steps

    const int num_bundles = 1;
    const int num_timesteps = 3;
    const int nsd = 7;
    const int num_sigma = 2 * nsd + 1;

    View3D r("r", num_bundles, num_timesteps, 3);
    View3D v("v", num_bundles, num_timesteps, 3);
    View2D m("m", num_bundles, num_timesteps);
    View4D sigmas("sigmas", num_bundles, num_sigma, nsd, num_timesteps);

    auto r_host = Kokkos::create_mirror_view(r);
    auto v_host = Kokkos::create_mirror_view(v);
    auto m_host = Kokkos::create_mirror_view(m);

    for (int t = 0; t < num_timesteps; ++t) {
        const auto& row = bundle_32_rows[t];
        for (int j = 0; j < 3; ++j) {
            r_host(0, t, j) = row(1 + j);  // x, y, z
            v_host(0, t, j) = row(4 + j);  // vx, vy, vz
        }
        m_host(0, t) = row(7);  // mass
    }

    Kokkos::deep_copy(r, r_host);
    Kokkos::deep_copy(v, v_host);
    Kokkos::deep_copy(m, m_host);

    std::vector<int> time_steps = {0, 1, 2};  // assume first 3 entries

    Eigen::MatrixXd P_pos = 1e-4 * Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd P_vel = 1e-6 * Eigen::MatrixXd::Identity(3, 3);
    double P_mass = 1e-8;

    double alpha = 1e-3, beta = 2.0, kappa = 0.0;

    generate_sigma_points_kokkos(
        nsd, alpha, beta, kappa,
        P_pos, P_vel, P_mass,
        time_steps, r, v, m, sigmas
    );

    auto sigmas_host = Kokkos::create_mirror_view(sigmas);
    Kokkos::deep_copy(sigmas_host, sigmas);

    SECTION("Sigma point shape and values") {
        REQUIRE(sigmas_host.extent(0) == num_bundles);
        REQUIRE(sigmas_host.extent(1) == num_sigma);
        REQUIRE(sigmas_host.extent(2) == nsd);
        REQUIRE(sigmas_host.extent(3) == num_timesteps);

        for (int t = 0; t < num_timesteps; ++t) {
            for (int d = 0; d < nsd; ++d) {
                double center_val = sigmas_host(0, 0, d, t);
                double avg = 0.0;
                for (int i = 0; i < num_sigma; ++i) {
                    avg += sigmas_host(0, i, d, t);
                }
                avg /= num_sigma;

                // Check that center sigma point is close to the average
                CHECK_THAT(center_val, WithinAbs(avg, 1e-3));
            }
        }
    }
}
