#include "../include/l1_dot_2B_propul_gpu.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

KOKKOS_INLINE_FUNCTION
void l1_dot_2B_propul(
    double* lam_dot_out,
    double F, double G, double H, double K, double L, double P, double T, double g0,
    double l_F, double l_G, double l_H, double l_K, double l_L, double l_P,
    double m, double m0, double mu
) {
    double sinL = sin(L);
    double cosL = cos(L);
    double w = 1.0 + F * cosL + G * sinL;
    double s = 1.0 + H * H + K * K;
    double sqrtP_mu = sqrt(P / mu);
    double w2 = w * w;
    double p32 = pow(P, 1.5);
    double mu32 = sqrt(mu) * sqrt(mu);

    // Compute common partial derivatives
    double dw_dF = cosL;
    double dw_dG = sinL;
    double dw_dL = -F * sinL + G * cosL;
    double ds_dH = 2.0 * H;
    double ds_dK = 2.0 * K;

    double dC1_dP = 0.5 * sqrt(1.0 / (P * mu));

    // Thrust direction magnitude (approximate, from symbolic model)
    double thrust_factor = T / (m0 * m * g0);

    // lam_dot_out calculations (based on simplified dependency)
    lam_dot_out[0] = 0.0; // dp_dot lambda
    lam_dot_out[1] = 0.0; // df_dot lambda
    lam_dot_out[2] = 0.0; // dg_dot lambda
    lam_dot_out[3] = 0.0; // dh_dot lambda
    lam_dot_out[4] = 0.0; // dk_dot lambda
    lam_dot_out[5] = 0.0; // dL_dot lambda

    // --- Fill in approximate expressions based on your Python ---
    lam_dot_out[0] = l_L * (-sqrt(mu) * (w2 / (2.0 * p32))) 
                     + l_F * thrust_factor * cosL
                     + l_G * thrust_factor * sinL;

    lam_dot_out[1] = l_L * (sqrt(mu) * (w / (P * sqrt(P))) * cosL)
                     - l_F * thrust_factor * (1.0 / w)
                     + l_G * thrust_factor * (dw_dF * w - dw_dG * cosL) / (w * w);

    lam_dot_out[2] = l_L * (sqrt(mu) * (w / (P * sqrt(P))) * sinL)
                     - l_F * thrust_factor * (dw_dG * w - dw_dF * sinL) / (w * w)
                     - l_G * thrust_factor * (1.0 / w);

    lam_dot_out[3] = -l_H * thrust_factor * s * cosL / (2.0 * w)
                     - l_K * thrust_factor * s * sinL / (2.0 * w);

    lam_dot_out[4] = -l_H * thrust_factor * s * sinL / (2.0 * w)
                     + l_K * thrust_factor * s * cosL / (2.0 * w);

    lam_dot_out[5] = l_F * thrust_factor * (H * sinL - K * cosL) / w
                     - l_G * thrust_factor * (H * cosL + K * sinL) / w
                     - l_H * thrust_factor * (H * H - K * K) * cosL / (2.0 * w)
                     - l_K * thrust_factor * (H * H - K * K) * sinL / (2.0 * w);
}
