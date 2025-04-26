#ifndef ODEFUNC_GPU_HPP
#define ODEFUNC_GPU_HPP

#include <Kokkos_Core.hpp>
#include "propagation_settings.hpp" 
#include "../include/l1_dot_2B_propul_gpu.hpp"
#include "../include/lm_dot_2B_propul_gpu.hpp"
#include <cmath>

KOKKOS_INLINE_FUNCTION
void odefunc(
    const double* x,
    double* dx,
    double t,
    const PropagationSettings& settings
) {
    const double p = x[0], f = x[1], g = x[2], h = x[3], k = x[4], L = x[5], m = x[6];
    const double lam_p = x[7], lam_f = x[8], lam_g = x[9];
    const double lam_h = x[10], lam_k = x[11], lam_L = x[12], lam_m = x[13];

    const double SinL = sin(L);
    const double CosL = cos(L);
    const double w = 1.0 + f * CosL + g * SinL;

    if (fabs(w) < 1e-10) {
        for (int i = 0; i < 14; ++i) {
            dx[i] = 0.0;
        }
        return;
    }

    const double s = 1.0 + h * h + k * k;
    const double C1 = sqrt(p / settings.mu);
    const double C2 = 1.0 / w;
    const double C3 = h * SinL - k * CosL;

    double A[6][3] = {
        {0.0,            2.0 * p * C2 * C1,        0.0},
        {C1 * SinL,      C1 * C2 * ((w + 1.0) * CosL + f),   -C1 * (g / w) * C3},
        {-C1 * CosL,     C1 * C2 * ((w + 1.0) * SinL + g),    C1 * (f / w) * C3},
        {0.0,            0.0,                     C1 * s * CosL * C2 / 2.0},
        {0.0,            0.0,                     C1 * s * SinL * C2 / 2.0},
        {0.0,            0.0,                     C1 * C2 * C3}
    };

    double b[6] = {0.0, 0.0, 0.0, 0.0, 0.0, sqrt(settings.mu) * (w * w / pow(p, 1.5))};

    double lam[6] = {lam_p, lam_f, lam_g, lam_h, lam_k, lam_L};

    // thrust_dir = A^T * lam
    double thrust_dir[3] = {0.0, 0.0, 0.0};
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 6; ++i) {
            thrust_dir[j] += A[i][j] * lam[i];
        }
    }

    // Normalize thrust_dir
    double norm_thrust_dir = sqrt(thrust_dir[0]*thrust_dir[0] + thrust_dir[1]*thrust_dir[1] + thrust_dir[2]*thrust_dir[2]);
    thrust_dir[0] /= norm_thrust_dir;
    thrust_dir[1] /= norm_thrust_dir;
    thrust_dir[2] /= norm_thrust_dir;

    // xdot = b + A * thrust_dir * (F / (m0 * m * g0))
    for (int i = 0; i < 6; ++i) {
        dx[i] = b[i];
        for (int j = 0; j < 3; ++j) {
            dx[i] += A[i][j] * thrust_dir[j] * (settings.F / (settings.m0 * m * settings.g0));
        }
    }

    // mdot
    dx[6] = -settings.F / (settings.m0 * settings.c);

    // Define capital letter versions for l1 and lm functions
    const double F = f;
    const double G = g;
    const double H = h;
    const double K = k;
    const double P = p;

    // lam_dot
    double lam_dot[6];
    l1_dot_2B_propul(
        lam_dot,
        F, G, H, K, L, P, settings.F, settings.g0,
        lam_f, lam_g, lam_h, lam_k, lam_L, lam_p,
        m, settings.m0, settings.mu
    );

    for (int i = 0; i < 6; ++i) {
        dx[7+i] = lam_dot[i];
    }

    // lam_m_dot
    dx[13] = lm_dot_2B_propul(
        F, G, H, K, L, P, settings.F, settings.g0,
        lam_f, lam_g, lam_h, lam_k, lam_L, lam_p,
        m, settings.m0, settings.mu
    );
}

#endif // ODEFUNC_GPU_HPP
