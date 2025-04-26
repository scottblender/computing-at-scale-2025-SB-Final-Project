#ifndef ODEFUNC_GPU_HPP
#define ODEFUNC_GPU_HPP

#include <Kokkos_Core.hpp>
#include "propagation_settings.hpp"  // make sure PropagationSettings struct is available

KOKKOS_INLINE_FUNCTION
void odefunc(
    const double* x,        // input state, size 14
    double* dx,             // output derivative, size 14
    double t,               // current time
    const PropagationSettings& settings  // settings (mu, F, c, m0, g0)
);

#endif // ODEFUNC_GPU_HPP
