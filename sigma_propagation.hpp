#ifndef SIGMA_PROPAGATION_HPP
#define SIGMA_PROPAGATION_HPP

#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <functional>

// Settings for the propagation
struct PropagationSettings {
    double mu;             // Gravitational parameter
    double F;              // Thrust magnitude
    double c;              // Exhaust velocity
    double m0;             // Reference mass
    double g0;             // Standard gravity
    int num_updates;       // [optional] unused here, may apply in other logic
    int num_eval_per_step; // Number of integration steps per time interval
    int state_size;        // Typically 7 (x, y, z, vx, vy, vz, m)
    int control_size;      // Typically 7 (lam0 through lam6)
};

// RK45 integrator with history output: returns [dim x steps+1] matrix
Eigen::MatrixXd rk45_integrate_history(
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&, double)> ode,
    const Eigen::VectorXd& state0,
    double t0, double t1,
    int steps
);

// Host-only propagation routine (uses Eigen for sampling, stores to Kokkos)
void propagate_sigma_trajectories(
    const Kokkos::View<double****>& sigmas_combined,   // [bundle][2n+1][7][time_steps]
    const Kokkos::View<double***>& new_lam_bundles,    // [time][7][bundle]
    const std::vector<double>& time,                   // key time points
    const std::vector<double>& Wm,                     // UKF weights (mean)
    const std::vector<double>& Wc,                     // UKF weights (cov)
    const PropagationSettings& settings,
    Kokkos::View<double****>& trajectories_out         // [bundle][2n+1][step][8]
);

#endif // SIGMA_PROPAGATION_HPP
