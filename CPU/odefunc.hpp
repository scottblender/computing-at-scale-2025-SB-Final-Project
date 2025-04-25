#ifndef ODEFUNC_HPP
#define ODEFUNC_HPP

#include <Eigen/Dense>

/**
 * Computes the derivative of the full 14D state vector for trajectory propagation.
 * 
 * @param t   Current time [s]
 * @param x   State vector: [p, f, g, h, k, L, m, lam_p, lam_f, lam_g, lam_h, lam_k, lam_L, lam_m]
 * @param dx  Output derivative vector (same shape as x)
 * @param mu  Gravitational parameter
 * @param F   Thrust magnitude
 * @param c   Specific impulse
 * @param m0  Initial mass
 * @param g0  Standard gravity
 */
void odefunc(
    double t,
    const Eigen::VectorXd& x,
    Eigen::VectorXd& dx,
    double mu,
    double F,
    double c,
    double m0,
    double g0
);

#endif // ODEFUNC_HPP
