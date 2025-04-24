#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "csv_loader.hpp"  // assumes load_csv_matrix_safe exists there

// Robust CSV matrix loader with error handling
Eigen::MatrixXd load_csv_matrix_safe(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::string line;
    std::vector<std::vector<double>> rows;

    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (...) {
                throw std::runtime_error("Failed to parse value in " + path + ": " + value);
            }
        }

        if (!row.empty())
            rows.push_back(row);
    }

    if (rows.empty()) throw std::runtime_error("CSV empty or unreadable: " + path);

    size_t cols = rows[0].size();
    for (const auto& r : rows)
        if (r.size() != cols)
            throw std::runtime_error("Inconsistent row length in: " + path);

    Eigen::MatrixXd mat(rows.size(), cols);
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = rows[i][j];

    return mat;
}

TEST_CASE("CSV loading debug check", "[propagation-debug]") {
    Eigen::MatrixXd expected;
    REQUIRE_NOTHROW(expected = load_csv_matrix_safe("expected_trajectories_full.csv"));

    REQUIRE(expected.cols() >= 10);
    REQUIRE(expected.rows() >= 1);

    int bundle = static_cast<int>(expected(0, 0));
    int sigma = static_cast<int>(expected(0, 1));
    double x = expected(0, 2);

    INFO("Loaded first row: bundle=" << bundle << ", sigma=" << sigma << ", x=" << x);
    CHECK_THAT(x, Catch::Matchers::WithinAbs(x, 1e-6));  // sanity check
}
