#ifndef CSV_LOADER_HPP
#define CSV_LOADER_HPP

#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>

using View4D = Kokkos::View<double****>;
using View3D = Kokkos::View<double***>;
using View2D = Kokkos::View<double**>;

// Load CSV into vector of vectors (expects specific number of columns)
inline std::vector<std::vector<double>> load_csv(const std::string& path, int expected_cols) {
    std::vector<std::vector<double>> data;
    std::ifstream file(path);
    std::string line;

    std::getline(file, line);  // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        if (row.size() == expected_cols)
            data.push_back(row);
    }
    return data;
}

// Load CSV into Eigen::MatrixXd
inline Eigen::MatrixXd load_csv_matrix(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    std::vector<std::vector<double>> rows;

    std::getline(file, line);  // Skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (...) {
                row.push_back(0.0);
            }
        }
        if (!row.empty()) rows.push_back(row);
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

// Load weights Wm, Wc from CSV: [sigma_index, Wm, Wc]
inline void load_weights(const std::string& path, std::vector<double>& Wm, std::vector<double>& Wc) {
    auto data = load_csv(path, 3);
    Wm.resize(data.size());
    Wc.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        Wm[i] = data[i][1];
        Wc[i] = data[i][2];
    }
}

// Load lam control data into Kokkos View (from initial_bundle_32.csv format)
inline void load_controls(const std::string& path, View3D& new_lam_bundles, int num_steps, int num_bundles) {
    auto data = load_csv(path, 17);  // [time, x, y, z, vx, vy, vz, mass, lam0..lam6, bundle_index]
    Kokkos::resize(new_lam_bundles, num_steps, 7, num_bundles);

    // Track how many entries we've seen for each bundle
    std::unordered_map<int, int> bundle_time_index;

    for (const auto& row : data) {
        int b = static_cast<int>(row[16]);
        int t = bundle_time_index[b];  // current time index for bundle b
        for (int j = 0; j < 7; ++j)
            new_lam_bundles(t, j, b) = row[9 + j];
        bundle_time_index[b]++;
    }
}


// Load sigma point state history from expected_trajectories_full.csv
inline void load_sigma_points(
    const std::string& path,
    View4D& sigmas_combined,
    std::vector<double>& time,
    int& num_bundles,
    int& num_sigma,
    int& num_steps
) {
    auto data = load_csv(path, 17);  // [bundle, sigma, x, y, z, vx, vy, vz, m, lam0..lam6, time]

    std::unordered_map<int, std::unordered_map<int, std::vector<std::vector<double>>>> bundle_map;
    std::unordered_map<int, std::vector<double>> time_map;

    for (const auto& row : data) {
        int b = static_cast<int>(row[0]);
        int s = static_cast<int>(row[1]);
        double t = row[16];
        std::vector<double> state(row.begin() + 2, row.begin() + 9);  // x..m
        bundle_map[b][s].push_back(state);
        time_map[b].push_back(t);
    }

    num_bundles = bundle_map.size();
    num_sigma = bundle_map.begin()->second.size();
    num_steps = bundle_map.begin()->second.begin()->second.size();

    Kokkos::resize(sigmas_combined, num_bundles, num_sigma, 7, num_steps);
    time.resize(num_steps);

    int b_idx = 0;
    for (const auto& [b, sigma_map] : bundle_map) {
        int s_idx = 0;
        for (const auto& [s, states] : sigma_map) {
            for (int t = 0; t < num_steps; ++t) {
                for (int j = 0; j < 7; ++j)
                    sigmas_combined(b_idx, s_idx, j, t) = states[t][j];
                if (s_idx == 0 && b_idx == 0)
                    time[t] = time_map[b][t];
            }
            ++s_idx;
        }
        ++b_idx;
    }
}

#endif  // CSV_LOADER_HPP
