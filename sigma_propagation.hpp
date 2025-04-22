#ifndef SIGMA_PROPAGATION_HPP
#define SIGMA_PROPAGATION_HPP

#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <vector>

// Settings for the propagation
struct PropagationSettings {
    double mu;
    double F;
    double c;
    double m0;
    double g0;
    int num_updates;
    int num_eval_per_step;
    int state_size;
    int control_size;
};

// RK45 integrator with history output
void rk45_integrate_history(
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&, double)> ode,
    const Eigen::VectorXd& state0,
    double t0, double t1,
    int steps,
    std::vector<Eigen::VectorXd>& state_history
);

// Main propagation routine
void propagate_sigma_trajectories(
    const Kokkos::View<double****>& sigmas_combined,   // [bundle][2n+1][7][time_steps]
    const Kokkos::View<double***>& new_lam_bundles,    // [time][7][bundle]
    const std::vector<double>& time,                   // key time points
    const std::vector<double>& Wm,                     // UKF weights (mean)
    const std::vector<double>& Wc,                     // UKF weights (cov)
    const PropagationSettings& settings,
    Kokkos::View<double****>& trajectories_out         // [bundle][2n+1][step][7]
);

#endif // SIGMA_PROPAGATION_HPP
