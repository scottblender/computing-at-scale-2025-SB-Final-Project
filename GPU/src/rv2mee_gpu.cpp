#include "../include/rv2mee_gpu.hpp"
#include <cmath>

KOKKOS_INLINE_FUNCTION
void rv2mee(
    const double* r_eci,
    const double* v_eci,
    double mu,
    double* mee_out
) {
    // Norm of r
    double radius = sqrt(r_eci[0]*r_eci[0] + r_eci[1]*r_eci[1] + r_eci[2]*r_eci[2]);

    // h = r × v
    double h_vec[3] = {
        r_eci[1] * v_eci[2] - r_eci[2] * v_eci[1],
        r_eci[2] * v_eci[0] - r_eci[0] * v_eci[2],
        r_eci[0] * v_eci[1] - r_eci[1] * v_eci[0]
    };
    double hmag = sqrt(h_vec[0]*h_vec[0] + h_vec[1]*h_vec[1] + h_vec[2]*h_vec[2]);
    double p = hmag * hmag / mu;

    // h_hat = h / |h|
    double h_hat[3] = { h_vec[0]/hmag, h_vec[1]/hmag, h_vec[2]/hmag };

    double denom = 1.0 + h_hat[2];
    double h_elem = -h_hat[1] / denom;
    double k_elem =  h_hat[0] / denom;

    // Construct f_hat and g_hat
    double ssqrd = 1.0 + k_elem * k_elem + h_elem * h_elem;

    double f_hat[3] = {
        (1.0 - k_elem * k_elem + h_elem * h_elem) / ssqrd,
        (2.0 * k_elem * h_elem) / ssqrd,
        (-2.0 * k_elem) / ssqrd
    };

    double g_hat[3] = {
        f_hat[1],
        (1.0 + k_elem * k_elem - h_elem * h_elem) / ssqrd,
        (2.0 * h_elem) / ssqrd
    };

    // e vector
    double cross_vh[3] = {
        v_eci[1] * h_vec[2] - v_eci[2] * h_vec[1],
        v_eci[2] * h_vec[0] - v_eci[0] * h_vec[2],
        v_eci[0] * h_vec[1] - v_eci[1] * h_vec[0]
    };

    double r_unit[3] = { r_eci[0] / radius, r_eci[1] / radius, r_eci[2] / radius };

    double e_vec[3] = {
        -r_unit[0] + cross_vh[0] / mu,
        -r_unit[1] + cross_vh[1] / mu,
        -r_unit[2] + cross_vh[2] / mu
    };

    // uhat = r̂, vhat = transverse unit vector
    double rdotv = r_eci[0]*v_eci[0] + r_eci[1]*v_eci[1] + r_eci[2]*v_eci[2];

    double vhat[3] = {
        (v_eci[0]*radius - rdotv * r_eci[0] / radius) / hmag,
        (v_eci[1]*radius - rdotv * r_eci[1] / radius) / hmag,
        (v_eci[2]*radius - rdotv * r_eci[2] / radius) / hmag
    };

    // Compute f, g
    double f = e_vec[0]*f_hat[0] + e_vec[1]*f_hat[1] + e_vec[2]*f_hat[2];
    double g = e_vec[0]*g_hat[0] + e_vec[1]*g_hat[1] + e_vec[2]*g_hat[2];

    // Compute cos(l) and sin(l)
    double cosl = r_unit[0] + vhat[1];
    double sinl = r_unit[1] - vhat[0];
    double l_nonmod = atan2(sinl, cosl);

    // Final MEE output
    mee_out[0] = p;
    mee_out[1] = f;
    mee_out[2] = g;
    mee_out[3] = h_elem;
    mee_out[4] = k_elem;
    mee_out[5] = l_nonmod;
}
