#!/bin/bash

set -e  # Exit immediately if any command fails

dir="$1"
build_type="$2"
compiler="$3"
backend="${4:-SERIAL}"  # Default backend if not provided

# Install locations (same as GitHub Actions cache)
KOKKOS_INSTALL="$HOME/deps/kokkos/install"
CATCH2_INSTALL="$HOME/deps/catch2/install"
EIGEN_INSTALL="$HOME/deps/eigen/install"

mkdir -p "$HOME/deps"

# --------------------------
# Build Kokkos (if not cached)
# --------------------------
if [ ! -d "$KOKKOS_INSTALL" ]; then
  echo "[INFO] Building Kokkos for backend: $backend"
  rm -rf "$dir/kokkos" "$dir/build-kokkos"
  git clone --depth 1 https://github.com/kokkos/kokkos "$dir/kokkos"

  kokkos_backend_flags="-DKokkos_ENABLE_SERIAL=ON"  # Default
  if [[ "$backend" == "CUDA" ]]; then
    kokkos_backend_flags="-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_VOLTA70=ON"
  elif [[ "$backend" == "OPENMP" ]]; then
    kokkos_backend_flags="-DKokkos_ENABLE_OPENMP=ON"
  fi

  cmake -S "$dir/kokkos" -B "$dir/build-kokkos" \
    -DCMAKE_INSTALL_PREFIX="$KOKKOS_INSTALL" \
    $kokkos_backend_flags \
    -DKokkos_ENABLE_DEPRECATED_CODE=OFF \
    -DKokkos_ENABLE_TESTS=OFF \
    -DKokkos_ENABLE_EXAMPLES=OFF \
    -DCMAKE_CXX_COMPILER="$compiler"

  cmake --build "$dir/build-kokkos" -j8 --target install
else
  echo "[INFO] Using cached Kokkos install"
fi

export CMAKE_PREFIX_PATH="$KOKKOS_INSTALL"

# --------------------------
# Build Catch2 (if not cached)
# --------------------------
if [ ! -d "$CATCH2_INSTALL" ]; then
  echo "[INFO] Building Catch2..."
  rm -rf "$dir/Catch2" "$dir/build-Catch2"
  git clone --depth 1 https://github.com/catchorg/Catch2 "$dir/Catch2"

  cmake -S "$dir/Catch2" -B "$dir/build-Catch2" \
    -DCMAKE_INSTALL_PREFIX="$CATCH2_INSTALL"
  cmake --build "$dir/build-Catch2" -j8 --target install
else
  echo "[INFO] Using cached Catch2 install"
fi

export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CATCH2_INSTALL"

# --------------------------
# Build Eigen (if not cached)
# --------------------------
if [ ! -d "$EIGEN_INSTALL" ]; then
  echo "[INFO] Installing Eigen (header-only)..."
  rm -rf "$dir/eigen" "$dir/build-eigen"
  git clone --depth 1 https://gitlab.com/libeigen/eigen.git "$dir/eigen"

  cmake -S "$dir/eigen" -B "$dir/build-eigen" \
    -DCMAKE_INSTALL_PREFIX="$EIGEN_INSTALL"
  cmake --build "$dir/build-eigen" --target install
else
  echo "[INFO] Using cached Eigen install"
fi

export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$EIGEN_INSTALL"

# --------------------------
# Clean & Build CPU project
# --------------------------
echo "[INFO] Building CPU version..."
rm -rf "$dir/CPU/build"
cmake -S . -B "$dir/CPU/build" \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DCMAKE_CXX_COMPILER="$compiler"
cmake --build "$dir/CPU/build" -j8

# --------------------------
# Clean & Build GPU project
# --------------------------
echo "[INFO] Building GPU version..."
rm -rf "$dir/GPU/build"
cmake -S . -B "$dir/GPU/build" \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DCMAKE_CXX_COMPILER="$compiler"
cmake --build "$dir/GPU/build" -j8

# --------------------------
# Copy CSV files for tests
# --------------------------
echo "[INFO] Copying test CSV files into CPU and GPU build directories..."
for f in expected_trajectories_bundle_32.csv initial_bundles_32_33.csv sigma_weights.csv; do
  if [ -f "$f" ]; then
    cp "$f" "$dir/CPU/build/"
    cp "$f" "$dir/GPU/build/"
  else
    echo "[ERROR] Missing required CSV file: $f"
    exit 1
  fi
done

echo "[INFO] Install completed successfully."
