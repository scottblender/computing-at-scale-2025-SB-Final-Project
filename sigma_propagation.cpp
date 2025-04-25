#include <Kokkos_Core.hpp> // ✅ Add this FIRST
#include "sigma_propagation.hpp" // ✅ Then include your header

#include "rv2mee.hpp"
#include "mee2rv.hpp"
#include "odefunc.hpp"
#include "l1_dot_2B_propul.hpp"
#include "lm_dot_2B_propul.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <functional>
#include <iostream>

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

    auto traj_host = Kokkos::create_mirror_view(trajectories_out);

    const int num_subintervals = 10;
    const int evals_per_subinterval = settings.num_eval_per_step / num_subintervals;
    Eigen::MatrixXd P = 0.001 * Eigen::MatrixXd::Identity(7, 7);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(P);
    Eigen::MatrixXd transform = solver.eigenvectors() *
                                solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();

    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);

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

                // Initial lambda at beginning of interval
                Eigen::VectorXd lam(7);
                for (int k = 0; k < 7; ++k)
                    lam(k) = lam_host(j, k, i);
                S.tail(7) = lam;

                auto ode = [&](const Eigen::VectorXd& x, Eigen::VectorXd& dxdt, double t) {
                    odefunc(t, x, dxdt, settings.mu, settings.F, settings.c, settings.m0, settings.g0);
                };

                double t_start = time[j];
                double t_end = time[j + 1];
                double total_h = (t_end - t_start) / settings.num_eval_per_step;
                int output_index = 0;

                for (int sub = 0; sub < num_subintervals; ++sub) {
                    double t0 = t_start + sub * (t_end - t_start) / num_subintervals;
                    double t1 = t_start + (sub + 1) * (t_end - t_start) / num_subintervals;

                    // Resample lambda except for first subinterval
                    if (sub > 0) {
                        Eigen::VectorXd z(7);
                        for (int k = 0; k < 7; ++k)
                            z(k) = dist(gen);
                        lam = lam + transform * z;
                        S.tail(7) = lam;
                    }

                    Eigen::MatrixXd history = rk45_integrate_history(ode, S, t0, t1, evals_per_subinterval);

                    for (int n = 0; n < history.cols(); ++n) {
                        Eigen::VectorXd state_n = history.col(n);
                        Eigen::Vector3d r_out, v_out;
                        mee2rv(state_n.head(6), settings.mu, r_out, v_out);

                        int index = j * (settings.num_eval_per_step + 1) + output_index++;
                        for (int k = 0; k < 3; ++k) {
                            traj_host(i, sigma_idx, index, k)     = r_out(k);
                            traj_host(i, sigma_idx, index, k + 3) = v_out(k);
                        }
                        traj_host(i, sigma_idx, index, 6) = state_n(6);
                        traj_host(i, sigma_idx, index, 7) = t0 + total_h * n;
                    }

                    // Update S and lambda for next subinterval
                    S = history.col(history.cols() - 1);
                    lam = S.tail(7);
                }
            }
        }
    }

    Kokkos::deep_copy(trajectories_out, traj_host);
}
