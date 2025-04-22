#include "sigma_points_kokkos.hpp"

struct SigmaPointFunctor {
    View3D r_bundles, v_bundles;
    View2D m_bundles;
    View4D sigmas_out;
    ViewMatrixHost L_host;
    int nsd;
    double scaling_factor;
    std::vector<int> time_steps;

    SigmaPointFunctor(
        const View3D& r_b, const View3D& v_b, const View2D& m_b,
        const View4D& sig_out, const Eigen::MatrixXd& L_eigen,
        int nsd_, double scale, const std::vector<int>& t_steps)
        : r_bundles(r_b), v_bundles(v_b), m_bundles(m_b),
          sigmas_out(sig_out), nsd(nsd_), scaling_factor(scale),
          time_steps(t_steps)
    {
        L_host = ViewMatrixHost("L_host", nsd_, nsd_);
        for (int i = 0; i < nsd_; ++i)
            for (int j = 0; j < nsd_; ++j)
                L_host(i, j) = L_eigen(i, j);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        for (size_t j = 0; j < time_steps.size(); ++j) {
            int t = time_steps[j];
            double mu[7];

            for (int k = 0; k < 3; ++k) mu[k] = r_bundles(i, t, k);
            for (int k = 0; k < 3; ++k) mu[3 + k] = v_bundles(i, t, k);
            mu[6] = m_bundles(i, t);

            for (int k = 0; k < nsd; ++k)
                sigmas_out(i, 0, k, j) = mu[k];

            for (int k = 0; k < nsd; ++k) {
                double offset[7];
                for (int d = 0; d < nsd; ++d)
                    offset[d] = scaling_factor * L_host(d, k);

                for (int d = 0; d < nsd; ++d) {
                    sigmas_out(i, 1 + k, d, j)       = mu[d] + offset[d];
                    sigmas_out(i, 1 + nsd + k, d, j) = mu[d] - offset[d];
                }
            }
        }
    }
};

void generate_sigma_points_kokkos(
    int nsd,
    double alpha,
    double beta,
    double kappa,
    const Eigen::MatrixXd& P_pos,
    const Eigen::MatrixXd& P_vel,
    double P_mass,
    const std::vector<int>& time_steps,
    const View3D& r_bundles,
    const View3D& v_bundles,
    const View2D& m_bundles,
    const View4D& sigmas_out
) {
    double lambda = alpha * alpha * (nsd + kappa) - nsd;
    double scaling_factor = std::sqrt(nsd + lambda);

    // Construct full covariance matrix
    Eigen::MatrixXd P_combined = Eigen::MatrixXd::Zero(nsd, nsd);
    P_combined.block(0, 0, 3, 3) = P_pos;
    P_combined.block(3, 3, 3, 3) = P_vel;
    P_combined(6, 6) = P_mass;

    // Cholesky
    Eigen::LLT<Eigen::MatrixXd> llt(P_combined);
    Eigen::MatrixXd L = llt.matrixL();

    // Launch Kokkos functor
    SigmaPointFunctor functor(
        r_bundles, v_bundles, m_bundles,
        sigmas_out, L, nsd, scaling_factor, time_steps
    );

    Kokkos::parallel_for("GenerateSigmaPoints", r_bundles.extent(0), functor);
    Kokkos::fence(); // Ensure completion
}
