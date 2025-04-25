#ifndef LM_DOT_2B_PROPUL_GPU_HPP
#define LM_DOT_2B_PROPUL_GPU_HPP

#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
double lm_dot_2B_propul(
    double T, double F, double G, double H, double K, double L,
    double P, double thrust, double g0,
    double l_F, double l_G, double l_H, double l_K,
    double l_L, double l_P, double m, double m0, double mu
);

#endif // LM_DOT_2B_PROPUL_GPU_HPP
