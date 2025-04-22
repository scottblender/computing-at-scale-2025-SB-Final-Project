#ifndef MEE2RV_HPP
#define MEE2RV_HPP

#include <Eigen/Dense>
#include <vector>

// Converts Modified Equinoctial Elements (MEE) to Cartesian position and velocity vectors.
// All inputs are assumed to be vectors of the same size N.
void mee2rv(
    const Eigen::VectorXd& p,  // Semi-latus rectum
    const Eigen::VectorXd& f,  // Equinoctial element f
    const Eigen::VectorXd& g,  // Equinoctial element g
    const Eigen::VectorXd& h,  // Equinoctial element h
    const Eigen::VectorXd& k,  // Equinoctial element k
    const Eigen::VectorXd& L,  // True longitude
    double mu,                 // Gravitational parameter
    std::vector<Eigen::Vector3d>& r_eci, // Output: position vectors (ECI frame)
    std::vector<Eigen::Vector3d>& v_eci  // Output: velocity vectors (ECI frame)
);

#endif  // MEE2RV_HPP
