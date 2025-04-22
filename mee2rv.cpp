#include "mee2rv.hpp"
#include <cmath>

void mee2rv(
    const Eigen::VectorXd& p,
    const Eigen::VectorXd& f,
    const Eigen::VectorXd& g,
    const Eigen::VectorXd& h,
    const Eigen::VectorXd& k,
    const Eigen::VectorXd& L,
    double mu,
    std::vector<Eigen::Vector3d>& r_eci,
    std::vector<Eigen::Vector3d>& v_eci
) {
    int N = L.size();
    r_eci.resize(N);
    v_eci.resize(N);

    for (int i = 0; i < N; ++i) {
        double cosL = std::cos(L[i]);
        double sinL = std::sin(L[i]);

        double radius = p[i] / (1 + f[i] * cosL + g[i] * sinL);
        double alpha2 = h[i]*h[i] - k[i]*k[i];
        double hk2 = h[i]*h[i] + k[i]*k[i];
        double s2 = 1.0 + hk2;

        // Position
        r_eci  = radius * ((cosL + alpha2 * cosL + 2 * h[i] * k[i] * sinL) / s2);
        r_eci  = radius * ((sinL - alpha2 * sinL + 2 * h[i] * k[i] * cosL) / s2);
        r_eci  = 2 * radius * ((h[i] * sinL - k[i] * cosL) / s2);

        // Velocity
        double sqrt_mu_over_p = std::sqrt(mu / p[i]);
        v_eci  = -sqrt_mu_over_p * ((sinL + alpha2 * sinL - 2 * h[i] * k[i] * cosL + g[i]
                          - 2 * f[i] * h[i] * k[i] + alpha2 * g[i]) / s2);

        v_eci  = -sqrt_mu_over_p * ((-cosL + alpha2 * cosL + 2 * h[i] * k[i] * sinL - f[i]
                          + 2 * g[i] * h[i] * k[i] + alpha2 * f[i]) / s2);

        v_eci  = 2 * sqrt_mu_over_p * ((h[i] * cosL + k[i] * sinL + f[i] * h[i] + g[i] * k[i]) / s2);
    }
}
