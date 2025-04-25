#include "sigma_propagation.hpp"
#include "rv2mee.hpp"
#include "mee2rv.hpp"
#include "odefunc.hpp"
#include "l1_dot_2B_propul.hpp"
#include "lm_dot_2B_propul.hpp"

#include <Eigen/Dense>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <random>
#include <functional>
#include <iostream>

// Sample from multivariate normal on host
Eigen::MatrixXd sample_controls_host(
    const Kokkos::View<double***>::HostMirror& mean_controls,
    int num_steps, int num_bundles
) {
    Eigen::MatrixXd samples(num_steps * num_bundles, 7);
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    Eigen::MatrixXd P = 0.001 * Eigen::MatrixXd::Identity(7, 7);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(P);
    Eigen::MatrixXd transform = solver.eigenvectors() *
                                solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();

    for (int b = 0; b < num_bundles; ++b) {
        for (int t = 0; t < num_steps; ++t) {
            Eigen::VectorXd mean(7), z(7);
            for (int i = 0; i < 7; ++i) {
                mean(i) = mean_controls(t, i, b);
                z(i) = dist(gen);
            }
            samples.row(t + b * num_steps) = (mean + transform * z).transpose();
        }
    }
    return samples;
}

// RK45 integration with history
Eigen::MatrixXd rk45_integrate_history(
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&, double)> ode,
    const Eigen::VectorXd& state0,
    double t0, double t1,
    int steps
) {
    int dim = state0.size();
    Eigen::MatrixXd history(dim, steps + 1);
    Eigen::VectorXd x = state0;
    history.col(0) = x;
    double h = (t1 - t0) / steps;
    double t = t0;

    Eigen::VectorXd k1(dim), k2(dim), k3(dim), k4(dim), k5(dim), k6(dim), dx(dim);

    for (int i = 0; i < steps; ++i) {
        ode(x, k1, t);
        ode(x + 0.25 * h * k1, k2, t + 0.25 * h);
        ode(x + (3.0 / 32.0) * h * k1 + (9.0 / 32.0) * h * k2, k3, t + (3.0 / 8.0) * h);
        ode(x + (1932.0 / 2197.0) * h * k1 - (7200.0 / 2197.0) * h * k2 + (7296.0 / 2197.0) * h * k3, k4, t + (12.0 / 13.0) * h);
        ode(x + (439.0 / 216.0) * h * k1 - 8.0 * h * k2 + (3680.0 / 513.0) * h * k3 - (845.0 / 4104.0) * h * k4, k5, t + h);
        ode(x - (8.0 / 27.0) * h * k1 + 2.0 * h * k2 - (3544.0 / 2565.0) * h * k3 + (1859.0 / 4104.0) * h * k4 - (11.0 / 40.0) * h * k5, k6, t + 0.5 * h);

        dx = (16.0 / 135.0) * k1 + (6656.0 / 12825.0) * k3 + (28561.0 / 56430.0) * k4 - (9.0 / 50.0) * k5 + (2.0 / 55.0) * k6;
        x += h * dx;
        t += h;
        history.col(i + 1) = x;
    }
    return history;
}

// Main propagation (host loop version with sampled controls)
void propagate_sigma_trajectories(
    const Kokkos::View<double****>& sigmas_combined,
    const Kokkos::View<double***>& new_lam_bundles,
    const std::vector<double>& time,
    const std::vector<double>& Wm,
    const std::vector<double>& Wc,
    const PropagationSettings& settings,
    Kokkos::View<double****>& trajectories_out
) {
    const int num_bundles = sigmas_combined.extent(0);
    const int num_sigma = sigmas_combined.extent(1);
    const int num_steps = sigmas_combined.extent(3);

    auto lam_host = Kokkos::create_mirror_view(new_lam_bundles);
    Kokkos::deep_copy(lam_host, new_lam_bundles);

    Eigen::MatrixXd sampled = sample_controls_host(lam_host, num_steps, num_bundles);

    auto traj_host = Kokkos::create_mirror_view(trajectories_out);

    for (int i = 0; i < num_bundles; ++i) {
        for (int sigma_idx = 0; sigma_idx < num_sigma; ++sigma_idx) {
            for (int j = 0; j < num_steps - 1; ++j) {
                Eigen::Vector3d r0, v0;
                for (int k = 0; k < 3; ++k) {
                    r0(k) = sigmas_combined(i, sigma_idx, k, j);
                    v0(k) = sigmas_combined(i, sigma_idx, k + 3, j);
                }
                double mass = sigmas_combined(i, sigma_idx, 6, j);
                Eigen::VectorXd mee = rv2mee(r0, v0, settings.mu);

                Eigen::VectorXd S(14);
                S.head(6) = mee;
                S(6) = mass;
                Eigen::VectorXd lam = sampled.row(j + i * num_steps);
                S.tail(7) = lam;

                auto ode = [&](const Eigen::VectorXd& x, Eigen::VectorXd& dxdt, double t) {
                    odefunc(t, x, dxdt, settings.mu, settings.F, settings.c, settings.m0, settings.g0);
                };

                Eigen::MatrixXd history = rk45_integrate_history(ode, S, time[j], time[j + 1], settings.num_eval_per_step);

                for (int n = 0; n <= settings.num_eval_per_step; ++n) {
                    Eigen::VectorXd state_n = history.col(n);
                    Eigen::Vector3d r_out, v_out;
                    mee2rv(state_n.head(6), settings.mu, r_out, v_out);

                    int index = j * (settings.num_eval_per_step + 1) + n;
                    for (int k = 0; k < 3; ++k) {
                        traj_host(i, sigma_idx, index, k)     = r_out(k);
                        traj_host(i, sigma_idx, index, k + 3) = v_out(k);
                    }
                    traj_host(i, sigma_idx, index, 6) = state_n(6);
                    traj_host(i, sigma_idx, index, 7) = time[j] + (time[j + 1] - time[j]) * (static_cast<double>(n) / settings.num_eval_per_step);
                }
            }
        }
    }

    Kokkos::deep_copy(trajectories_out, traj_host);
}
