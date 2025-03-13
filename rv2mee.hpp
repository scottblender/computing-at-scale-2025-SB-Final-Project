#ifndef RV2MEE_H
#define RV2MEE_H

#include <Eigen/Dense>

// Function to convert position and velocity vectors (ECI frame) to Equinoctial Orbital Elements
Eigen::VectorXd rv2mee(const Eigen::Vector3d& r_eci, const Eigen::Vector3d& v_eci, double mu);

#endif // RV2MEE_H
