#include "../include/odefunc_gpu.hpp"
#include "../include/l1_dot_2B_propul_gpu.hpp"
#include "../include/lm_dot_2B_propul_gpu.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

void odefunc(
    double t,
    const Eigen::VectorXd& x,
    Eigen::VectorXd& dx,
    double mu,
    double F,
    double c,
    double m0,
    double g0
) {
    const double p = x[0], f = x[1], g = x[2], h = x[3], k = x[4], L = x[5], m = x[6];
    const double lam_p = x[7], lam_f = x[8], lam_g = x[9];
    const double lam_h = x[10], lam_k = x[11], lam_L = x[12], lam_m = x[13];

    const double SinL = std::sin(L);
    const double CosL = std::cos(L);
    const double w = 1.0 + f * CosL + g * SinL;

    if (std::abs(w) < 1e-10) {
        dx = Eigen::VectorXd::Zero(x.size());
        return;
    }

    const double s = 1.0 + h * h + k * k;
    const double alpha = h * h - k * k;
    const double C1 = std::sqrt(p / mu);
    const double C2 = 1.0 / w;
    const double C3 = h * SinL - k * CosL;

    Eigen::MatrixXd A(6, 3);
    A << 0,            2 * p * C2 * C1,        0,
         C1 * SinL,    C1 * C2 * ((w + 1) * CosL + f),   -C1 * (g / w) * C3,
        -C1 * CosL,    C1 * C2 * ((w + 1) * SinL + g),    C1 * (f / w) * C3,
         0,            0,                     C1 * s * CosL * C2 / 2.0,
         0,            0,                     C1 * s * SinL * C2 / 2.0,
         0,            0,                     C1 * C2 * C3;

    Eigen::VectorXd b(6);
    b << 0, 0, 0, 0, 0, std::sqrt(mu) * (w * w / std::pow(p, 1.5));

    Eigen::MatrixXd lam(6, 1);
    lam << lam_p, lam_f, lam_g, lam_h, lam_k, lam_L;

    Eigen::VectorXd thrust_dir = A.transpose() * lam;
    thrust_dir.normalize();

    Eigen::VectorXd xdot = b + A * thrust_dir * (F / (m0 * m * g0));
    double mdot = -F / (m0 * c);

    Eigen::VectorXd lam_dot = l1_dot_2B_propul(F, f, g, h, k, L, p, F, g0,
                                               lam_f, lam_g, lam_h, lam_k, lam_L, lam_p, m, m0, mu);
    double lam_m_dot = lm_dot_2B_propul(F, f, g, h, k, L, p, F, g0,
                                        lam_f, lam_g, lam_h, lam_k, lam_L, lam_p, m, m0, mu);

    dx.resize(14);
    dx.head(6) = xdot;
    dx(6) = mdot;
    dx.segment(7, 6) = lam_dot;
    dx(13) = lam_m_dot;
}
