#ifndef SIGMA_PROPAGATION_HPP
#define SIGMA_PROPAGATION_HPP

#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <functional>

// Settings for the propagation
struct PropagationSettings {
    double mu;             // Gravitational parameter [non-dimensional]
    double F;              // Thrust magnitude
    double c;              // Exhaust velocity
    double m0;             // Reference mass
    double g0;             // Standard gravity
    int num_eval_per_step; // Total evaluations per [time_k, time_k+1] interval
    int state_size;        // Typically 7 (mee + mass)
    int control_size;      // Typically 7 (lam0 to lam6)

    // Optional: expose number of subintervals if needed externally
    // int num_subintervals = 10;
};

// RK45 integrator with history output: returns [dim x steps+1] matrix
Eigen::MatrixXd rk45_integrate_history(
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&, double)> ode,
    const Eigen::VectorXd& state0,
    double t0, double t1,
    int steps
);

// Host-only propagation routine (uses Eigen + Kokkos, resamples control each subinterval)
void propagate_sigma_trajectories(
    const Kokkos::View<double****>& sigmas_combined,   // [bundle][2n+1][7][time_steps]
    const Kokkos::View<double***>& new_lam_bundles,    // [time][7][bundle]
    const std::vector<double>& time,                   // time vector of length 2 (or more)
    const std::vector<double>& Wm,                     // UKF weights for mean
    const std::vector<double>& Wc,                     // UKF weights for covariance
    const PropagationSettings& settings,
    Kokkos::View<double****>& trajectories_out         // [bundle][2n+1][step][8] (x,y,z,vx,vy,vz,mass,time)
);

#endif // SIGMA_PROPAGATION_HPP
