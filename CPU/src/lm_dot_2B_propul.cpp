#include "../include/lm_dot_2B_propul.hpp"
#include <cmath>

double lm_dot_2B_propul(
    double T, double F, double G, double H, double K, double L,
    double P, double thrust, double g0,
    double l_F, double l_G, double l_H, double l_K,
    double l_L, double l_P, double m, double m0, double mu
) {
    double t2 = 1.0 / mu;
    double t3 = P * t2;
    double t4 = std::sqrt(t3);
    double t5 = t4;
    double t6 = std::sin(L);
    double t7 = std::cos(L);
    double t8 = H * t6;
    double t9 = F * t7;
    double t10 = G * t6;
    double t11 = t9 + t10 + 1.0;
    double t12 = 1.0 / t11;
    double t32 = K * t7;
    double t13 = t8 - t32;
    double t14 = F * t7 * 2.0;
    double t15 = G * t6 * 2.0;
    double t16 = t14 + t15 + 2.0;
    double t17 = 1.0 / t16;
    double t18 = H * H;
    double t19 = K * K;
    double t20 = t18 + t19 + 1.0;

    double t33 = l_H * t5 * t7 * t17 * t20;
    double t34 = G * l_F * t5 * t12 * t13;
    double t35 = l_K * t5 * t6 * t17 * t20;
    double t21 = std::abs(t33 - t34 + t35 + l_L * t5 * t12 * t13 + F * l_G * t5 * t12 * t13);

    double t22 = l_G * t5 * t7;
    double t37 = l_F * t5 * t6;
    double t23 = t22 - t37;

    double t29 = F + t7;
    double t30 = t12 * t29;
    double t31 = t7 + t30;

    double t39 = l_F * t5 * t31;
    double t40 = G + t6;
    double t41 = t12 * t40;
    double t42 = t6 + t41;

    double t43 = l_G * t5 * t42;

    return -(t21 + std::abs(t23) + t39 + t43);
}
