#include "sigma_propagation.hpp"
#include "rv2mee.hpp"
#include "mee2rv.hpp"
#include "odefunc.hpp"
#include "l1_dot_2B_propul.hpp"
#include "lm_dot_2B_propul.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <random>
#include <iostream>

void sample_controls_device(
    const Kokkos::View<double***>& mean_controls,
    Kokkos::View<double**>& samples,
    int num_steps,
    int num_bundles,
    Kokkos::Random_XorShift64_Pool<> rand_pool
) {
    Kokkos::parallel_for("sample_controls", num_steps * num_bundles, KOKKOS_LAMBDA(int idx) {
        int t = idx % num_steps;
        int b = idx / num_steps;

        auto rand_gen = rand_pool.get_state();
        for (int i = 0; i < 7; ++i) {
            double z = Kokkos::rand<Kokkos::Random_XorShift64_Pool<>, double>::draw(rand_gen);
            samples(t + b * num_steps, i) = mean_controls(t, i, b) + 0.001 * z;
        }
        rand_pool.free_state(rand_gen);
    });
}

// GPU-Compatible RK45 Integration
struct RK45Integrator {
    Kokkos::View<double**> history;
    Kokkos::View<double*> time_vec;

    RK45Integrator(Kokkos::View<double**> h, Kokkos::View<double*> t) : history(h), time_vec(t) {}

    KOKKOS_FUNCTION void operator()(const int i) const {
        double t = time_vec(i);
        double h = time_vec(i + 1) - time_vec(i);

        Kokkos::View<double*> x = history(i);
        Kokkos::View<double*> k1("k1", x.extent(0)), k2("k2", x.extent(0));
        Kokkos::View<double*> k3("k3", x.extent(0)), k4("k4", x.extent(0));
        Kokkos::View<double*> k5("k5", x.extent(0)), k6("k6", x.extent(0)), dx("dx", x.extent(0));

        // Apply ODE function in GPU-friendly form
        odefunc(t, x, k1);
        odefunc(t + 0.25 * h, x + 0.25 * h * k1, k2);
        odefunc(t + (3.0 / 8.0) * h, x + (3.0 / 32.0) * h * k1 + (9.0 / 32.0) * h * k2, k3);
        odefunc(t + (12.0 / 13.0) * h, x + (1932.0 / 2197.0) * h * k1 - (7200.0 / 2197.0) * h * k2 + (7296.0 / 2197.0) * h * k3, k4);
        odefunc(t + h, x + (439.0 / 216.0) * h * k1 - 8.0 * h * k2 + (3680.0 / 513.0) * h * k3 - (845.0 / 4104.0) * h * k4, k5);
        odefunc(t + 0.5 * h, x - (8.0 / 27.0) * h * k1 + 2.0 * h * k2 - (3544.0 / 2565.0) * h * k3 + (1859.0 / 4104.0) * h * k4 - (11.0 / 40.0) * h * k5, k6);

        dx = (16.0 / 135.0) * k1 + (6656.0 / 12825.0) * k3 + (28561.0 / 56430.0) * k4
           - (9.0 / 50.0) * k5 + (2.0 / 55.0) * k6;

        for (int j = 0; j < x.extent(0); ++j) {
            x(j) += h * dx(j);
        }
    }
};

// GPU-Compatible Sigma Propagation
void propagate_sigma_trajectories(
    const Kokkos::View<double****>& sigmas_combined,
    const Kokkos::View<double***>& new_lam_bundles,
    const std::vector<double>& time,
    const std::vector<double>& Wm,
    const std::vector<double>& Wc,
    const PropagationSettings& settings,
    Kokkos::View<double****>& trajectories_out
) {
    const int num_bundles = sigmas_combined.extent(0);
    const int num_sigma = sigmas_combined.extent(1);
    const int num_steps = sigmas_combined.extent(3);

    Kokkos::View<double***> lam_host("lam_host", num_steps, 7, num_bundles);
    Kokkos::deep_copy(lam_host, new_lam_bundles);

    RandomPool rand_pool(42);

    Kokkos::parallel_for("propagate_sigma", num_bundles * num_sigma * (num_steps - 1), KOKKOS_LAMBDA(int idx) {
        int i = idx / (num_sigma * (num_steps - 1));
        int sigma_idx = (idx / (num_steps - 1)) % num_sigma;
        int j = idx % (num_steps - 1);

        Kokkos::View<double*> S("S", 14);
        Kokkos::View<double*> lam("lam", 7);
        for (int k = 0; k < 7; ++k) {
            lam(k) = lam_host(j, k, i);
        }
        S.tail(7) = lam;

        Kokkos::View<double*> history("history", settings.num_eval_per_step, 14);
        Kokkos::View<double*> time_values("time_values", settings.num_eval_per_step);
        
        RK45Integrator rk45(history, time_values);
        Kokkos::parallel_for("rk45_steps", settings.num_eval_per_step, rk45);

        Kokkos::parallel_for("store_results", settings.num_eval_per_step, KOKKOS_LAMBDA(int n) {
            for (int k = 0; k < 8; ++k) {
                trajectories_out(i, sigma_idx, j * settings.num_eval_per_step + n, k) = history(n, k);
            }
        });
    });
}