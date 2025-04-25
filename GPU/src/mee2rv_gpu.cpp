#include "../include/mee2rv_gpu.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

KOKKOS_INLINE_FUNCTION
void mee2rv(
    const double* mee,
    double mu,
    double* r_eci,
    double* v_eci
) {
    double p = mee[0];
    double f = mee[1];
    double g = mee[2];
    double h = mee[3];
    double k = mee[4];
    double L = mee[5];

    double cosL = cos(L);
    double sinL = sin(L);

    double s2 = 1.0 + h * h + k * k;
    double w = 1.0 + f * cosL + g * sinL;
    double r = p / w;
    double sqrt_mu_over_p = sqrt(mu / p);
    double alpha2 = h * h - k * k;

    // Position
    r_eci[0] = r * (cosL + alpha2 * cosL + 2.0 * h * k * sinL) / s2;
    r_eci[1] = r * (sinL - alpha2 * sinL + 2.0 * h * k * cosL) / s2;
    r_eci[2] = 2.0 * r * (h * sinL - k * cosL) / s2;

    // Velocity
    v_eci[0] = -sqrt_mu_over_p * (
        sinL + alpha2 * sinL - 2.0 * h * k * cosL + g
        - 2.0 * f * h * k + alpha2 * g
    ) / s2;

    v_eci[1] = -sqrt_mu_over_p * (
        -cosL + alpha2 * cosL + 2.0 * h * k * sinL - f
        + 2.0 * g * h * k + alpha2 * f
    ) / s2;

    v_eci[2] = 2.0 * sqrt_mu_over_p * (
        h * cosL + k * sinL + f * h + g * k
    ) / s2;
}
