#include "../include/l1_dot_2B_propul.hpp"
#include <Eigen/Dense>
#include <cmath>

Eigen::VectorXd l1_dot_2B_propul(
    double T, double F, double G, double H, double K, double L,
    double P, double thrust, double g0,
    double l_F, double l_G, double l_H, double l_K,
    double l_L, double l_P, double m, double m0, double mu
) {
    double t2 = 1.0 / mu;
    double t3 = std::cos(L);
    double t4 = P * t2;
    double t5 = std::sqrt(t4);
    double t7 = 1.0 / t5;
    double t8 = std::sin(L);
    double t9 = H * t8;
    double t10 = F * t3;
    double t11 = G * t8;
    double t12 = t10 + t11 + 1.0;
    double t13 = 1.0 / t12;
    double t31 = K * t3;
    double t14 = t9 - t31;
    double t15 = F * t3 * 2.0;
    double t16 = G * t8 * 2.0;
    double t17 = t15 + t16 + 2.0;
    double t18 = 1.0 / t17;
    double t19 = H * H;
    double t20 = K * K;
    double t21 = t19 + t20 + 1.0;

    double t32 = l_H * t3 * t7 * t18 * t21;
    double t33 = G * l_F * t7 * t13 * t14;
    double t34 = l_K * t7 * t8 * t18 * t21;
    double t22 = std::abs(t32 - t33 + t34 + l_L * t7 * t13 * t14 + F * l_G * t7 * t13 * t14);

    double t36 = l_G * t3 * t7;
    double t37 = l_F * t7 * t8;
    double t23 = std::abs(t36 - t37);

    double t28 = G + t8;
    double t29 = t13 * t28;
    double t30 = t8 + t29;

    double t40 = F + t3;
    double t41 = t13 * t40;
    double t42 = t3 + t41;

    double t43 = l_F * t7 * t42;
    double t44 = l_G * t7 * t30;

    Eigen::VectorXd out(6);
    out << -t22, -t23, -t22, -t22, -t22, -(t43 + t44);

    return out;
}
