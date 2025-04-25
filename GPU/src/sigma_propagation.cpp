#include "sigma_propagation.hpp"
#include "mee2rv.hpp"
#include "rv2mee.hpp"
#include "odefunc.hpp"

KOKKOS_INLINE_FUNCTION
void rk45_step(
    void (*odefunc)(const double*, double*, double, const PropagationSettings&),
    const double* x0, double t0, double t1, int steps,
    const PropagationSettings& settings,
    double history[][14], double* time_out
) {
    double h = (t1 - t0) / (steps - 1);
    double x[14];
    for (int i = 0; i < 14; ++i) x[i] = x0[i];

    double k1[14], k2[14], k3[14], k4[14], k5[14], k6[14], dx[14];
    for (int s = 0; s < steps; ++s) {
        for (int i = 0; i < 14; ++i) history[s][i] = x[i];
        time_out[s] = t0 + s * h;

        odefunc(x, k1, t0, settings);

        double xtemp[14];
        for (int i = 0; i < 14; ++i) xtemp[i] = x[i] + 0.25 * h * k1[i];
        odefunc(xtemp, k2, t0 + 0.25 * h, settings);

        for (int i = 0; i < 14; ++i)
            xtemp[i] = x[i] + h * ((3.0/32.0)*k1[i] + (9.0/32.0)*k2[i]);
        odefunc(xtemp, k3, t0 + 0.375 * h, settings);

        for (int i = 0; i < 14; ++i)
            xtemp[i] = x[i] + h * ((1932.0/2197.0)*k1[i] - (7200.0/2197.0)*k2[i] + (7296.0/2197.0)*k3[i]);
        odefunc(xtemp, k4, t0 + (12.0/13.0) * h, settings);

        for (int i = 0; i < 14; ++i)
            xtemp[i] = x[i] + h * ((439.0/216.0)*k1[i] - 8.0*k2[i] + (3680.0/513.0)*k3[i] - (845.0/4104.0)*k4[i]);
        odefunc(xtemp, k5, t0 + h, settings);

        for (int i = 0; i < 14; ++i)
            xtemp[i] = x[i] - h * ((8.0/27.0)*k1[i] - 2.0*k2[i] + (3544.0/2565.0)*k3[i] - (1859.0/4104.0)*k4[i] + (11.0/40.0)*k5[i]);
        odefunc(xtemp, k6, t0 + 0.5 * h, settings);

        for (int i = 0; i < 14; ++i)
            dx[i] = h * ((16.0/135.0)*k1[i] + (6656.0/12825.0)*k3[i] + (28561.0/56430.0)*k4[i]
                       - (9.0/50.0)*k5[i] + (2.0/55.0)*k6[i]);

        for (int i = 0; i < 14; ++i) x[i] += dx[i];
    }
}

void propagate_sigma_trajectories(
    const View4D& sigmas_combined,
    const View3D& new_lam_bundles,
    const View1D& time,
    const View1D& Wm,
    const View1D& Wc,
    const View2D& random_controls,
    const View2D& transform,
    const PropagationSettings& settings,
    View4D& trajectories_out
) {
    auto lam_host = Kokkos::create_mirror_view(new_lam_bundles);
    auto traj_host = Kokkos::create_mirror_view(trajectories_out);
    auto time_host = Kokkos::create_mirror_view(time);
    auto rand_host = Kokkos::create_mirror_view(random_controls);
    auto transform_host = Kokkos::create_mirror_view(transform);

    Kokkos::deep_copy(lam_host, new_lam_bundles);
    Kokkos::deep_copy(time_host, time);
    Kokkos::deep_copy(rand_host, random_controls);
    Kokkos::deep_copy(transform_host, transform);

    int num_bundles = sigmas_combined.extent(0);
    int num_sigma = sigmas_combined.extent(1);
    int num_steps = sigmas_combined.extent(3);
    int num_sub = settings.num_subintervals;
    int evals = settings.num_eval_per_step / num_sub;

    int rand_idx = 0;

    for (int i = 0; i < num_bundles; ++i) {
        for (int sigma = 0; sigma < num_sigma; ++sigma) {
            for (int j = 0; j < num_steps - 1; ++j) {
                double r[3], v[3];
                for (int k = 0; k < 3; ++k) {
                    r[k] = sigmas_combined(i, sigma, k, j);
                    v[k] = sigmas_combined(i, sigma, k+3, j);
                }
                double mass = sigmas_combined(i, sigma, 6, j);

                double mee[6], dummy[3];
                rv2mee(r, v, settings.mu, mee, dummy);

                double state[14];
                for (int k = 0; k < 6; ++k) state[k] = mee[k];
                state[6] = mass;
                for (int k = 0; k < 7; ++k) state[7+k] = lam_host(j, k, i);

                int out_idx = 0;
                for (int sub = 0; sub < num_sub; ++sub) {
                    double dt = (time_host(j+1) - time_host(j)) / num_sub;
                    double t0 = time_host(j) + dt * sub;
                    double t1 = time_host(j) + dt * (sub + 1);

                    if (sub > 0) {
                        for (int k = 0; k < 7; ++k) {
                            double dz = 0.0;
                            for (int d = 0; d < 7; ++d)
                                dz += transform_host(d, k) * rand_host(rand_idx, d);
                            state[7 + k] += dz;
                        }
                        rand_idx++;
                    }

                    double history[200][14];
                    double tvals[200];
                    rk45_step(odefunc, state, t0, t1, evals, settings, history, tvals);

                    for (int n = 0; n < evals; ++n) {
                        double rout[3], vout[3];
                        mee2rv(&history[n][0], settings.mu, rout, vout);
                        int idx = j * settings.num_eval_per_step + out_idx++;
                        for (int k = 0; k < 3; ++k) {
                            traj_host(i, sigma, idx, k) = rout[k];
                            traj_host(i, sigma, idx, k+3) = vout[k];
                        }
                        traj_host(i, sigma, idx, 6) = history[n][6];
                        traj_host(i, sigma, idx, 7) = tvals[n];
                    }

                    for (int k = 0; k < 14; ++k)
                        state[k] = history[evals - 1][k];
                }
            }
        }
    }

    Kokkos::deep_copy(trajectories_out, traj_host);
}
