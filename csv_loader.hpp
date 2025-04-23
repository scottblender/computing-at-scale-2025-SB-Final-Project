#pragma once
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

// Utility: Load CSV into a vector of vectors
inline std::vector<std::vector<double>> load_csv(const std::string& path, int expected_cols) {
    std::vector<std::vector<double>> data;
    std::ifstream file(path);
    std::string line;
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

// Load Wm and Wc from weights CSV
inline void load_weights(const std::string& path, std::vector<double>& Wm, std::vector<double>& Wc) {
    auto data = load_csv(path, 3);
    Wm.resize(data.size());
    Wc.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        Wm[i] = data[i][1];
        Wc[i] = data[i][2];
    }
}

// Load control data into [time][7][bundle]
inline void load_controls(const std::string& path, View3D& new_lam_bundles, int num_steps, int num_bundles) {
    auto data = load_csv(path, 9);
    Kokkos::resize(new_lam_bundles, num_steps, 7, num_bundles);
    for (const auto& row : data) {
        int t = static_cast<int>(row[0]);
        int b = static_cast<int>(row[8]);
        for (int j = 0; j < 7; ++j)
            new_lam_bundles(t, j, b) = row[1 + j];
    }
}

// Load initial sigma state history into [bundle][sigma][7][step]
inline void load_sigma_points(
    const std::string& path,
    View4D& sigmas_combined,
    std::vector<double>& time,
    int& num_bundles,
    int& num_sigma,
    int& num_steps
) {
    auto data = load_csv(path, 10); // [t, x, y, z, vx, vy, vz, m, bundle, sigma]

    std::unordered_map<int, std::unordered_map<int, std::vector<std::vector<double>>>> bundle_map;
    std::unordered_map<int, std::vector<double>> time_map;

    for (const auto& row : data) {
        double t = row[0];
        int b = static_cast<int>(row[8]);
        int s = static_cast<int>(row[9]);
        std::vector<double> state(row.begin() + 1, row.begin() + 8);  // x..m
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
