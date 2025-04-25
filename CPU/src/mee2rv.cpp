#include "../include/mee2rv.hpp"
#include <cmath>

void mee2rv(
    const Eigen::VectorXd& mee, // not GPU-compatible
    double mu,
    Eigen::Vector3d& r_eci,
    Eigen::Vector3d& v_eci
) {
    double p = mee(0);
    double f = mee(1);
    double g = mee(2);
    double h = mee(3);
    double k = mee(4);
    double L = mee(5);

    double cosL = std::cos(L);
    double sinL = std::sin(L);

    double s2 = 1 + h * h + k * k;
    double w = 1 + f * cosL + g * sinL;
    double r = p / w;
    double sqrt_mu_over_p = std::sqrt(mu / p);
    double alpha2 = h * h - k * k;

    // Position
    r_eci(0) = r * (cosL + alpha2 * cosL + 2 * h * k * sinL) / s2;
    r_eci(1) = r * (sinL - alpha2 * sinL + 2 * h * k * cosL) / s2;
    r_eci(2) = 2 * r * (h * sinL - k * cosL) / s2;

    // Velocity
    v_eci(0) = -sqrt_mu_over_p * (
        sinL + alpha2 * sinL - 2 * h * k * cosL + g
        - 2 * f * h * k + alpha2 * g
    ) / s2;

    v_eci(1) = -sqrt_mu_over_p * (
        -cosL + alpha2 * cosL + 2 * h * k * sinL - f
        + 2 * g * h * k + alpha2 * f
    ) / s2;

    v_eci(2) = 2 * sqrt_mu_over_p * (
        h * cosL + k * sinL + f * h + g * k
    ) / s2;
}
