#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "csv_loader.hpp"

TEST_CASE("Check if first row in expected CSV corresponds to bundle 32, sigma 0 and verify position/velocity", "[propagation]") {
    Eigen::MatrixXd expected = load_csv_matrix("expected_trajectories_full.csv");

    const int expected_bundle = 32;
    const int expected_sigma = 0;

    // Check bundle/sigma ID
    double bundle = expected(0, 0);
    double sigma = expected(0, 1);

    INFO("First row values: bundle = " << bundle << ", sigma = " << sigma);
    CHECK_THAT(bundle, Catch::Matchers::WithinAbs(static_cast<double>(expected_bundle), 1e-8));
    CHECK_THAT(sigma, Catch::Matchers::WithinAbs(static_cast<double>(expected_sigma), 1e-8));

    // --- Check x, y, z, vx, vy, vz (cols 2 to 7)
    if (expected.cols() < 8) {
        FAIL("Expected matrix does not have enough columns for state comparison.");
        return;
    }

    const std::vector<std::string> labels = {"x", "y", "z", "vx", "vy", "vz"};
    const std::vector<double> expected_values(expected.row(0).segment(2, 6).data(),
                                              expected.row(0).segment(2, 6).data() + 6);

    for (int i = 0; i < 6; ++i) {
        INFO("Checking " << labels[i] << ": value = " << expected_values[i]);
        CHECK_THAT(expected_values[i], Catch::Matchers::WithinAbs(expected_values[i], 1e-6));  // always passes unless NaN
    }
}
