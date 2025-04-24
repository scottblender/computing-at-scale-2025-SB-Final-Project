#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>  
#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "csv_loader.hpp"

TEST_CASE("Check if first row in expected CSV corresponds to bundle 32, sigma 0", "[propagation]") {
    Eigen::MatrixXd expected = load_csv_matrix("expected_trajectories_full.csv");

    const int expected_bundle = 32;
    const int expected_sigma = 0;

    double bundle = expected(0, 0);
    double sigma = expected(0, 1);

    INFO("First row values: bundle = " << bundle << ", sigma = " << sigma);
    CHECK_THAT(bundle, Catch::Matchers::WithinAbs(static_cast<double>(expected_bundle), 1e-8));
    CHECK_THAT(sigma, Catch::Matchers::WithinAbs(static_cast<double>(expected_sigma), 1e-8));
}
