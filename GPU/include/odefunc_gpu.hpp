#ifndef ODEFUNC_GPU_HPP
#define ODEFUNC_GPU_HPP

#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
void odefunc(
    double t,
    const double* x,     // input state, size 14
    double* dx,          // output derivative, size 14
    double mu,
    double F,
    double c,
    double m0,
    double g0
);

#endif // ODEFUNC_GPU_HPP
