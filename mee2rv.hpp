#ifndef MEE2RV_HPP
#define MEE2RV_HPP

#include <Eigen/Dense>
#include <vector>

void mee2rv(
    const Eigen::VectorXd& p,
    const Eigen::VectorXd& f,
    const Eigen::VectorXd& g,
    const Eigen::VectorXd& h,
    const Eigen::VectorXd& k,
    const Eigen::VectorXd& L,
    double mu,
    std::vector<Eigen::Vector3d>& r_eci,
    std::vector<Eigen::Vector3d>& v_eci
);

#endif
