#ifndef SIGMA_PROPAGATION_HPP
#define SIGMA_PROPAGATION_HPP

#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <functional>

// Settings for the propagation
struct PropagationSettings {
    double mu;                  // Gravitational parameter [non-dimensional]
    double F;                   // Thrust magnitude
    double c;                   // Exhaust velocity
    double m0;                  // Reference mass
    double g0;                  // Standard gravity
    int num_eval_per_step;      // RK45 evaluations per [t_k, t_{k+1}] (yields num_eval_per_step + 1 total points)
    int num_subintervals = 10;  // Number of subintervals per [t_k, t_{k+1}] (control resampled each)
    int state_size;             // Size of state vector (usually 7: mee + mass)
    int control_size;           // Size of control vector (usually 7: lambda)
};

// RK45 integrator with history output: returns pair of [dim x (steps+1)] matrix and time values
std::pair<Eigen::MatrixXd, Eigen::VectorXd> rk45_integrate_history(
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&, double)> ode,
    const Eigen::VectorXd& state0,
    double t0, double t1,
    int steps
);

// Main propagation routine (host-only)
void propagate_sigma_trajectories(
    const Kokkos::View<double****>& sigmas_combined,   // [bundle][2n+1][7][time_steps]
    const Kokkos::View<double***>& new_lam_bundles,    // [time][7][bundle]
    const std::vector<double>& time,                   // Time vector of length >= 2
    const std::vector<double>& Wm,                     // UKF weights for mean
    const std::vector<double>& Wc,                     // UKF weights for covariance
    const PropagationSettings& settings,
    Kokkos::View<double****>& trajectories_out         // [bundle][2n+1][step][8]: (x,y,z,vx,vy,vz,mass,time)
);

// Optional: Sample control vectors from multivariate normal on host
Eigen::MatrixXd sample_controls_host(
    const Kokkos::View<double***>::HostMirror& mean_controls,
    int num_steps,
    int num_bundles
);

#endif // SIGMA_PROPAGATION_HPP
