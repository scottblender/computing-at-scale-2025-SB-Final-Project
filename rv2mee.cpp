#include "rv2mee.h"
#include <cmath>  // For M_PI constant and other math functions

// Function to convert position and velocity vectors (ECI frame) to Equinoctial Orbital Elements
Eigen::VectorXd rv2mee(const Eigen::Vector3d& r_eci, const Eigen::Vector3d& v_eci, double mu) {
    // Step 1: Compute the magnitude of the position vector
    double radius = r_eci.norm();  // r = ||r_eci||

    // Step 2: Calculate the angular momentum vector (cross product of r and v)
    Eigen::Vector3d h_vec = r_eci.cross(v_eci);  // h = r x v
    double hmag = h_vec.norm();  // hmag = ||h_vec|| (angular momentum magnitude)

    // Step 3: Calculate the semi-latus rectum (p) from angular momentum and gravitational parameter
    double p = std::pow(hmag, 2) / mu;  // p = h^2 / mu

    // Step 4: Compute the unit vector of angular momentum (h_hat)
    Eigen::Vector3d h_hat = h_vec / hmag;  // h_hat = h / ||h||

    // Step 5: Calculate the Equinoctial Elements K and H
    double denom = 1 + h_hat[2];  // denominator used to compute k and h
    double h_elem = -h_hat[1] / denom;  // h element
    double k_elem = h_hat[0] / denom;   // k element

    // Step 6: Construct the unit vectors (f_hat, g_hat) in the Equinoctial frame
    Eigen::Vector3d f_hat, g_hat;

    f_hat[0] = 1 - std::pow(k_elem, 2) + std::pow(h_elem, 2);  // f_hat[0] = 1 - k^2 + h^2
    f_hat[1] = 2 * k_elem * h_elem;  // f_hat[1] = 2kh
    f_hat[2] = -2 * k_elem;  // f_hat[2] = -2k

    g_hat[0] = f_hat[1];  // g_hat[0] = f_hat[1]
    g_hat[1] = 1 + std::pow(k_elem, 2) - std::pow(h_elem, 2);  // g_hat[1] = 1 + k^2 - h^2
    g_hat[2] = 2 * h_elem;  // g_hat[2] = 2h

    // Step 7: Compute additional parameters for further calculations
    double rdotv = r_eci.dot(v_eci);  // dot product of r_eci and v_eci (r.v)
    Eigen::Vector3d cross_vh = v_eci.cross(h_vec);  // cross product of v_eci and h_vec (v x h)
    double ssqrd = 1 + std::pow(k_elem, 2) + std::pow(h_elem, 2);  // denominator for normalization

    // Step 8: Normalize the unit vectors (f_hat and g_hat)
    f_hat /= ssqrd;  // normalize f_hat
    g_hat /= ssqrd;  // normalize g_hat

    // Step 9: Compute the eccentricity vector (e_vec)
    Eigen::Vector3d e_vec = (-r_eci / radius) + (cross_vh / mu);  // eccentricity vector

    // Step 10: Compute the unit vectors uhat and vhat
    Eigen::Vector3d uhat = r_eci / radius;  // uhat = r / radius
    Eigen::Vector3d vhat = (v_eci * radius - (rdotv / radius) * r_eci) / hmag;  // vhat calculation

    // Step 11: Compute the f and g equinoctial elements (dot products with e_vec)
    double g = e_vec.dot(g_hat);  // g = e_vec . g_hat
    double f = e_vec.dot(f_hat);  // f = e_vec . f_hat

    // Step 12: Compute the true longitude (l_nonmod) using geometry
    double cosl = uhat[0] + vhat[1];  // cosine of longitude
    double sinl = uhat[1] - vhat[0];  // sine of longitude
    double l_nonmod = std::atan2(sinl, cosl);  // true longitude (in radians)

    // Step 13: Assemble the final array of Equinoctial Elements
    Eigen::VectorXd mee(6);
    mee[0] = p;  // Semi-latus rectum
    mee[1] = f;  // First Equinoctial element
    mee[2] = g;  // Second Equinoctial element
    mee[3] = h_elem;  // Third Equinoctial element
    mee[4] = k_elem;  // Fourth Equinoctial element
    mee[5] = l_nonmod;  // True longitude (in radians)

    // Return the Equinoctial Elements as an Eigen vector
    return mee;
}
