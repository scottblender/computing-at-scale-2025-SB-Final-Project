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

// Externally implemented conversion + dynamics functions
Eigen::VectorXd rv2mee(const Eigen::Vector3d& r, const Eigen::Vector3d& v, double mu);
void mee2rv(const Eigen::VectorXd& mee, double mu, Eigen::Vector3d& r, Eigen::Vector3d& v);
void odefunc(double t, const Eigen::VectorXd& state, Eigen::VectorXd& dstate, double mu, double F, double c, double m0, double g0);

// Main propagation routine
void propagate_sigma_trajectories(
    const Kokkos::View<double****>& sigmas_combined,   // [bundle][2n+1][7][time_steps]
    const Kokkos::View<double***>& new_lam_bundles,    // [time][7][bundle]
    const std::vector<double>& time,                   // key time points
    const std::vector<double>& Wm,                     // UKF weights (mean)
    const std::vector<double>& Wc,                     // UKF weights (cov)
    const PropagationSettings& settings,
    Kokkos::View<double******>& trajectories_out       // [bundle][2n+1][step][7]
);

#endif // SIGMA_PROPAGATION_HPP