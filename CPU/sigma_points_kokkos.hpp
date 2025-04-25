#ifndef SIGMA_POINTS_KOKKOS_HPP
#define SIGMA_POINTS_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <vector>

using View3D = Kokkos::View<double***>;      // [bundle][timestep][dim]
using View2D = Kokkos::View<double**>;       // [bundle][timestep]
using View4D = Kokkos::View<double****>;     // [bundle][2n+1][dim][timestep]
using ViewMatrixHost = Kokkos::View<double**, Kokkos::HostSpace>; // Cholesky matrix on host

/**
 * Generates sigma points using Kokkos in parallel.
 * The sigma points are output as a 4D View of shape [bundle][2n+1][7][time_step].
 *
 * @param nsd Dimensionality of the state (should be 7 for pos/vel/mass)
 * @param alpha UKF spread parameter
 * @param beta UKF prior knowledge parameter
 * @param kappa UKF secondary spread parameter
 * @param P_pos 3x3 Eigen matrix for position covariance
 * @param P_vel 3x3 Eigen matrix for velocity covariance
 * @param P_mass Scalar variance of the mass
 * @param time_steps Indices of the time steps to sample (from backTspan)
 * @param r_bundles Kokkos View for position bundles
 * @param v_bundles Kokkos View for velocity bundles
 * @param m_bundles Kokkos View for mass bundles
 * @param sigmas_out Kokkos View to store the output sigma points
 */
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
);

#endif // SIGMA_POINTS_KOKKOS_HPP
