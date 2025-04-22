#!/bin/bash

dir=$1
build_type=$2
compiler=$3
backend=${4:-SERIAL}  # Default to SERIAL if not provided (options: SERIAL, CUDA, OPENMP)

# --------------------------
# Build Kokkos with selected backend
# --------------------------
rm -rf $dir/kokkos
git clone https://github.com/kokkos/kokkos $dir/kokkos

rm -rf $dir/build-kokkos

kokkos_backend_flags="-DKokkos_ENABLE_SERIAL=ON"

if [[ "$backend" == "CUDA" ]]; then
  kokkos_backend_flags="-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON"
elif [[ "$backend" == "OPENMP" ]]; then
  kokkos_backend_flags="-DKokkos_ENABLE_OPENMP=ON"
fi

cmake -S $dir/kokkos -B $dir/build-kokkos \
  -DCMAKE_INSTALL_PREFIX=$HOME/kokkos/install \
  $kokkos_backend_flags \
  -DKokkos_ENABLE_DEPRECATED_CODE=OFF

cmake --build $dir/build-kokkos -j8 --target install
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$HOME/kokkos/install

# --------------------------
# Build Catch2
# --------------------------
rm -rf $dir/Catch2
git clone https://github.com/catchorg/Catch2 $dir/Catch2

rm -rf $dir/build-Catch2
cmake -S $dir/Catch2 -B $dir/build-Catch2 \
  -DCMAKE_INSTALL_PREFIX=$dir/build-Catch2/install/
cmake --build $dir/build-Catch2 -j8 --target install

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$dir/build-Catch2/install

# --------------------------
# Build Eigen (header-only)
# --------------------------
rm -rf $dir/eigen
git clone --depth 1 https://gitlab.com/libeigen/eigen.git $dir/eigen

rm -rf $dir/build-eigen
cmake -S $dir/eigen -B $dir/build-eigen \
  -DCMAKE_INSTALL_PREFIX=$dir/build-eigen/install
cmake --build $dir/build-eigen --target install

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$dir/build-eigen/install
# --------------------------
# Build Project
# --------------------------
rm -rf $dir/build
cmake . -B $dir/build \
  -DCMAKE_BUILD_TYPE=$build_type \
  -DCMAKE_CXX_COMPILER=$compiler
cmake --build $dir/build -j8