#include "sigma_propagation.hpp"
#include <random>
#include <functional>

// Sample from multivariate normal
Eigen::VectorXd sample_control(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov) {
    static std::mt19937 gen(42);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    Eigen::MatrixXd transform = solver.eigenvectors() * solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    Eigen::VectorXd z(mean.size());
    for (int i = 0; i < z.size(); ++i)
        z(i) = std::normal_distribution<>(0.0, 1.0)(gen);
    return mean + transform * z;
}

// Basic RK45 (fixed-step)
void rk45_integrate(
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&, double)> ode,
    Eigen::VectorXd& state,
    double t0, double t1,
    int steps
) {
    double h = (t1 - t0) / steps;
    Eigen::VectorXd k1, k2, k3, k4, k5, k6, dx;
    Eigen::VectorXd x = state;
    double t = t0;

    for (int i = 0; i < steps; ++i) {
        ode(x, k1, t);
        ode(x + 0.25 * h * k1, k2, t + 0.25 * h);
        ode(x + (3.0/32.0)*h*k1 + (9.0/32.0)*h*k2, k3, t + (3.0/8.0)*h);
        ode(x + (1932.0/2197.0)*h*k1 - (7200.0/2197.0)*h*k2 + (7296.0/2197.0)*h*k3, k4, t + (12.0/13.0)*h);
        ode(x + (439.0/216.0)*h*k1 - 8*h*k2 + (3680.0/513.0)*h*k3 - (845.0/4104.0)*h*k4, k5, t + h);
        ode(x - (8.0/27.0)*h*k1 + 2*h*k2 - (3544.0/2565.0)*h*k3 + (1859.0/4104.0)*h*k4 - (11.0/40.0)*h*k5, k6, t + 0.5*h);

        dx = (16.0/135.0)*k1 + (6656.0/12825.0)*k3 + (28561.0/56430.0)*k4 - (9.0/50.0)*k5 + (2.0/55.0)*k6;
        x += h * dx;
        t += h;
    }
    state = x;
}

// Main Kokkos-parallel propagation function
void propagate_sigma_trajectories(
    const Kokkos::View<double****>& sigmas_combined,
    const Kokkos::View<double***>& new_lam_bundles,
    const std::vector<double>& time,
    const std::vector<double>& Wm,
    const std::vector<double>& Wc,
    const PropagationSettings& settings,
    Kokkos::View<double******>& trajectories_out
) {
    const int num_bundles = sigmas_combined.extent(0);
    const int num_sigma = sigmas_combined.extent(1);
    const int num_steps = sigmas_combined.extent(3);

    Kokkos::parallel_for("propagate_all_bundles", num_bundles, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < num_steps - 1; ++j) {
            for (int sigma_idx = 0; sigma_idx < num_sigma; ++sigma_idx) {
                // Extract position, velocity, mass
                Eigen::Vector3d r0, v0;
                for (int k = 0; k < 3; ++k) {
                    r0(k) = sigmas_combined(i, sigma_idx, k, j);
                    v0(k) = sigmas_combined(i, sigma_idx, k + 3, j);
                }
                double mass = sigmas_combined(i, sigma_idx, 6, j);

                // Convert to MEE
                Eigen::VectorXd mee = rv2mee(r0, v0, settings.mu);

                // Combine MEE + mass + control
                Eigen::VectorXd S(14);
                S.head(6) = mee;
                S(6) = mass;
                for (int k = 0; k < 7; ++k)
                    S(7 + k) = new_lam_bundles(j, k, i);

                // Sample control
                Eigen::MatrixXd P_lam = 0.001 * Eigen::MatrixXd::Identity(7, 7);
                S.tail(7) = sample_control(S.tail(7), P_lam);

                // Define ode wrapper
                auto ode = [&](const Eigen::VectorXd& x, Eigen::VectorXd& dxdt, double t) {
                    odefunc(t, x, dxdt, settings.mu, settings.F, settings.c, settings.m0, settings.g0);
                };

                // Integrate
                rk45_integrate(ode, S, time[j], time[j + 1], settings.num_eval_per_step);

                // Convert back to RV
                Eigen::Vector3d r_out, v_out;
                mee2rv(S.head(6), settings.mu, r_out, v_out);

                // Store
                for (int k = 0; k < 3; ++k) {
                    trajectories_out(i, sigma_idx, j, k)     = r_out(k);
                    trajectories_out(i, sigma_idx, j, k + 3) = v_out(k);
                }
                trajectories_out(i, sigma_idx, j, 6) = S(6);
            }
        }
    });
}
