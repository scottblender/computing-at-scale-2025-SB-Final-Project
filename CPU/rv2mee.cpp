#include "rv2mee.hpp"
#include <cmath>

Eigen::VectorXd rv2mee(const Eigen::Vector3d& r_eci, const Eigen::Vector3d& v_eci, double mu) {
    double radius = r_eci.norm();
    Eigen::Vector3d h_vec = r_eci.cross(v_eci);
    double hmag = h_vec.norm();
    double p = std::pow(hmag, 2) / mu;

    Eigen::Vector3d h_hat = h_vec / hmag;
    double denom = 1 + h_hat[2];
    double h_elem = -h_hat[1] / denom;
    double k_elem = h_hat[0] / denom;

    Eigen::Vector3d f_hat, g_hat;
    f_hat[0] = 1 - k_elem * k_elem + h_elem * h_elem;
    f_hat[1] = 2 * k_elem * h_elem;
    f_hat[2] = -2 * k_elem;

    g_hat[0] = f_hat[1];
    g_hat[1] = 1 + k_elem * k_elem - h_elem * h_elem;
    g_hat[2] = 2 * h_elem;

    double rdotv = r_eci.dot(v_eci);
    Eigen::Vector3d cross_vh = v_eci.cross(h_vec);
    double ssqrd = 1 + k_elem * k_elem + h_elem * h_elem;
    f_hat /= ssqrd;
    g_hat /= ssqrd;

    Eigen::Vector3d e_vec = (-r_eci / radius) + (cross_vh / mu);
    Eigen::Vector3d uhat = r_eci / radius;
    Eigen::Vector3d vhat = (v_eci * radius - (rdotv / radius) * r_eci) / hmag;

    double g = e_vec.dot(g_hat);
    double f = e_vec.dot(f_hat);

    double cosl = uhat[0] + vhat[1];
    double sinl = uhat[1] - vhat[0];
    double l_nonmod = std::atan2(sinl, cosl);

    Eigen::VectorXd mee(6);
    mee[0] = p;
    mee[1] = f;
    mee[2] = g;
    mee[3] = h_elem;
    mee[4] = k_elem;
    mee[5] = l_nonmod;

    return mee; // not GPU-compatible
}
