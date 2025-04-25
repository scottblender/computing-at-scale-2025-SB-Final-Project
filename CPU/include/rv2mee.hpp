#ifndef RV2MEE_HPP
#define RV2MEE_HPP

#include <Eigen/Dense>

/**
 * @brief Converts position and velocity vectors (ECI) to Modified Equinoctial Elements (MEE).
 *
 * @param r_eci Position vector in ECI coordinates.
 * @param v_eci Velocity vector in ECI coordinates.
 * @param mu    Gravitational parameter.
 * @return      Vector of MEE elements [p, f, g, h, k, L].
 */
Eigen::VectorXd rv2mee(const Eigen::Vector3d& r_eci, const Eigen::Vector3d& v_eci, double mu);

#endif // RV2MEE_HPP
