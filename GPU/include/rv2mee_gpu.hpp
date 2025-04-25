#ifndef RV2MEE_GPU_HPP
#define RV2MEE_GPU_HPP

#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
void rv2mee(
    const double* r_eci,    // input position vector (3)
    const double* v_eci,    // input velocity vector (3)
    double mu,
    double* mee_out         // output MEE vector (6)
);

#endif // RV2MEE_GPU_HPP
