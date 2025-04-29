#ifndef KOKKOS_TYPES_HPP
#define KOKKOS_TYPES_HPP

#include <Kokkos_Core.hpp>
// Conditionally define memory space based on CUDA availability
#ifdef KOKKOS_ENABLE_CUDA
    #define MEMORY_SPACE Kokkos::CudaSpace
#else
    #define MEMORY_SPACE Kokkos::HostSpace
#endif

// Typedefs for convenience with MEMORY_SPACE
using View4D = Kokkos::View<double****, MEMORY_SPACE>;
using View3D = Kokkos::View<double***, MEMORY_SPACE>;
using View2D = Kokkos::View<double**, MEMORY_SPACE>;
using View1D = Kokkos::View<double*, MEMORY_SPACE>;

// Common typedefs
using DeviceMatrix = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>;
using Device4D = Kokkos::View<double****, Kokkos::DefaultExecutionSpace::memory_space>;
using HostMatrix = Kokkos::View<double**, Kokkos::HostSpace>;

#endif // KOKKOS_TYPES_HPP
