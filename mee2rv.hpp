#ifndef MEE2RV_HPP
#define MEE2RV_HPP

#include <Eigen/Dense>

// Single MEE-to-RV conversion
void mee2rv(
    const Eigen::VectorXd& mee,
    double mu,
    Eigen::Vector3d& r_eci,
    Eigen::Vector3d& v_eci
);

#endif
