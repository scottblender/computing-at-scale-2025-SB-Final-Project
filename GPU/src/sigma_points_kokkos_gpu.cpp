#include "../include/sigma_points_kokkos_gpu.hpp"
#include <cmath>

struct SigmaPointFunctor {
    View3D r_bundles, v_bundles;
    View2D m_bundles;
    View4D sigmas_out;
    ViewMatrixHost L_host;
    Kokkos::View<int*> time_steps;
    int nsd;
    double scaling_factor;

    SigmaPointFunctor(
        const View3D& r_b, const View3D& v_b, const View2D& m_b,
        const View4D& sig_out, const ViewMatrixHost& L_in,
        int nsd_, double scale, const Kokkos::View<int*>& t_steps)
        : r_bundles(r_b), v_bundles(v_b), m_bundles(m_b),
          sigmas_out(sig_out), L_host(L_in), nsd(nsd_),
          scaling_factor(scale), time_steps(t_steps)
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        for (int j = 0; j < time_steps.extent(0); ++j) {
            int t = time_steps(j);
            double mu[7];

            for (int k = 0; k < 3; ++k) mu[k] = r_bundles(i, t, k);
            for (int k = 0; k < 3; ++k) mu[3+k] = v_bundles(i, t, k);
            mu[6] = m_bundles(i, t);

            for (int k = 0; k < nsd; ++k)
                sigmas_out(i, 0, k, j) = mu[k];

            for (int k = 0; k < nsd; ++k) {
                double offset[7];
                for (int d = 0; d < nsd; ++d)
                    offset[d] = scaling_factor * L_host(d, k);

                for (int d = 0; d < nsd; ++d) {
                    sigmas_out(i, 1+k, d, j)       = mu[d] + offset[d];
                    sigmas_out(i, 1+nsd+k, d, j) = mu[d] - offset[d];
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
    const double* P_pos_flat, // length 9
    const double* P_vel_flat, // length 9
    double P_mass,
    const Kokkos::View<int*> time_steps,
    const View3D& r_bundles,
    const View3D& v_bundles,
    const View2D& m_bundles,
    const View4D& sigmas_out
) {
    double lambda = alpha * alpha * (nsd + kappa) - nsd;
    double scaling_factor = std::sqrt(nsd + lambda);

    // Host-only covariance assembly
    ViewMatrixHost P_combined("P_combined", nsd, nsd);

    for (int i = 0; i < 3; ++i) // pos
        for (int j = 0; j < 3; ++j)
            P_combined(i, j) = P_pos_flat[i*3 + j];

    for (int i = 0; i < 3; ++i) // vel
        for (int j = 0; j < 3; ++j)
            P_combined(i+3, j+3) = P_vel_flat[i*3 + j];

    P_combined(6, 6) = P_mass;

    // Manual Cholesky decomposition (lower triangular)
    for (int j = 0; j < nsd; ++j) {
        for (int i = j; i < nsd; ++i) {
            double sum = P_combined(i,j);
            for (int k = 0; k < j; ++k)
                sum -= P_combined(i,k) * P_combined(j,k);
            if (i == j) {
                P_combined(i,j) = sqrt(sum);
            } else {
                P_combined(i,j) = sum / P_combined(j,j);
            }
        }
        for (int k = j+1; k < nsd; ++k) {
            P_combined(j,k) = 0.0; // upper triangle zero
        }
    }

    // Launch Kokkos Functor
    SigmaPointFunctor functor(
        r_bundles, v_bundles, m_bundles,
        sigmas_out, P_combined, nsd, scaling_factor, time_steps
    );

    Kokkos::parallel_for("GenerateSigmaPoints", r_bundles.extent(0), functor);
    Kokkos::fence(); // Ensure completion
}
