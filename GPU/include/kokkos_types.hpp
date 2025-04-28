#ifndef KOKKOS_TYPES_HPP
#define KOKKOS_TYPES_HPP

#include <Kokkos_Core.hpp>

// Common typedefs
using DeviceMatrix = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space>;
using Device4D = Kokkos::View<double****, Kokkos::DefaultExecutionSpace::memory_space>;
using HostMatrix = Kokkos::View<double**, Kokkos::HostSpace>;

#endif // KOKKOS_TYPES_HPP
