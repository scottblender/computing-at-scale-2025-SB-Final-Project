#ifndef L1_DOT_2B_PROPUL_HPP
#define L1_DOT_2B_PROPUL_HPP

#include <Eigen/Dense>

Eigen::VectorXd l1_dot_2B_propul(
    double T, double f, double g, double h, double k, double L,
    double p, double F, double g0,
    double lam_f, double lam_g, double lam_h, double lam_k,
    double lam_L, double lam_p, double m, double m0, double mu
);

#endif // L1_DOT_2B_PROPUL_HPP
