#ifndef MEE2RV_HPP
#define MEE2RV_HPP

#include <Eigen/Dense>
#include <vector>

/**
 * @brief Converts modified equinoctial elements (MEE) to position and velocity vectors in ECI frame.
 *
 * @param p     Vector of semi-latus rectum values.
 * @param f     Vector of f equinoctial elements.
 * @param g     Vector of g equinoctial elements.
 * @param h     Vector of h equinoctial elements.
 * @param k     Vector of k equinoctial elements.
 * @param L     Vector of true longitudes (radians).
 * @param mu    Gravitational parameter.
 * @param r_eci Output vector of position vectors.
 * @param v_eci Output vector of velocity vectors.
 */
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

#endif // MEE2RV_HPP
