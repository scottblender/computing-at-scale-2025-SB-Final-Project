#ifndef CSV_LOADER_GPU_HPP
#define CSV_LOADER_GPU_HPP

#include <Kokkos_Core.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>

using View4D = Kokkos::View<double****>;
using View3D = Kokkos::View<double***>;
using View2D = Kokkos::View<double**>;
using View1D = Kokkos::View<double*>;

// Load generic CSV to vector<vector<double>> (HOST SIDE)
inline std::vector<std::vector<double>> load_csv(const std::string& path, int expected_cols) {
    std::vector<std::vector<double>> data;
    std::ifstream file(path);
    std::string line;
    if (!file.is_open())
        throw std::runtime_error("Unable to open CSV: " + path);

    std::getline(file, line);  // Skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                row.push_back(0.0); // replace bad values with 0.0
            }
        }
        if (row.size() == expected_cols)
            data.push_back(row);
    }
    return data;
}

// Load weights Wm and Wc from CSV
inline void load_weights(const std::string& path, std::vector<double>& Wm, std::vector<double>& Wc) {
    auto data = load_csv(path, 3);
    Wm.resize(data.size());
    Wc.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        Wm[i] = data[i][1];
        Wc[i] = data[i][2];
    }
}

// Load lam control values into a Kokkos View
inline void load_controls(const std::string& path, View3D& new_lam_bundles, int num_steps, int num_bundles) {
    auto data = load_csv(path, 17);  // [time, x,y,z,vx,vy,vz,m,lam0..lam6,bundle]
    Kokkos::resize(new_lam_bundles, num_steps, 7, num_bundles);
    for (size_t i = 0; i < data.size(); ++i) {
        int t = static_cast<int>(i);  // assuming sorted by time
        int b = static_cast<int>(data[i][16]); // bundle index
        for (int j = 0; j < 7; ++j) {
            new_lam_bundles(t, j, b) = data[i][9 + j];
        }
    }
}

// Load sigma points states into View4D
inline void load_sigma_points(
    const std::string& path,
    View4D& sigmas_combined,
    std::vector<double>& time,
    int& num_bundles,
    int& num_sigma,
    int& num_steps
) {
    auto data = load_csv(path, 17); // [bundle, sigma, x,y,z,vx,vy,vz,m,lam0..lam6,time]

    // Organize data into [bundle][sigma][steps]
    std::unordered_map<int, std::unordered_map<int, std::vector<std::vector<double>>>> bundle_map;
    std::unordered_map<int, std::vector<double>> time_map;

    for (const auto& row : data) {
        int b = static_cast<int>(row[0]);
        int s = static_cast<int>(row[1]);
        double t = row[16];
        std::vector<double> state(row.begin() + 2, row.begin() + 9);  // x,y,z,vx,vy,vz,m
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

#endif // CSV_LOADER_GPU_HPP
