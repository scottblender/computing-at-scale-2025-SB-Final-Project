#ifndef MEE2RV_GPU_HPP
#define MEE2RV_GPU_HPP

#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
void mee2rv(
    const double* mee,  // input MEE elements, length 6
    double mu,
    double* r_eci,      // output Cartesian position, length 3
    double* v_eci       // output Cartesian velocity, length 3
);

#endif // MEE2RV_GPU_HPP
