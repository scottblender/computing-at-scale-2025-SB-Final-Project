# Computing at Scale 2025 – SB Final Project

This project implements parallel sigma point trajectory propagation using [Kokkos](https://github.com/kokkos/kokkos) for execution portability across **Serial**, **OpenMP**, and **CUDA** backends. It supports trajectory sampling, RK45-based propagation, and performance benchmarking.

## 🔧 Dependencies

- CMake ≥ 3.20.0
- C++17-compatible compiler (e.g., `g++`, `clang++`, or `nvcc` with CUDA)
- Python (for CSV generation, optional)
- [Kokkos](https://github.com/kokkos/kokkos) (header-only or installed)
- Catch2 (for tests)
- Eigen

## 📦 Build Instructions

Clone the repository:

```bash
git clone https://github.com/scottblender/computing-at-scale-2025-SB-Final-Project.git
cd computing-at-scale-2025-SB-Final-Project
```

### 🧵 Serial Backend

```bash
./install.sh RelWithDebInfo SERIAL
```

### 🧵 OpenMP Backend

```bash
./install.sh RelWithDebInfo OPENMP
```

Ensure OpenMP is supported by your compiler (`g++ -fopenmp`).

### 🚀 CUDA Backend

```bash
./install.sh RelWithDebInfo CUDA
```

> Make sure to replace `Kokkos_ARCH_VOLTA70` with your GPU's architecture (e.g., `AMPERE80`, `KEPLER35`, etc.)

---

## 🧪 Running Tests

To run GPU-compatible or serial propagation tests:

```bash
cd GPU/build
./test_sigma_propagation_gpu
```

To run CPU-only tests:

```bash
cd CPU/build
./test_sigma_propagation_cpu
```

---

## 📈 Benchmarking Runtime vs Timesteps

Run benchmark executable to measure performance:

```bash
cd GPU/build
./runtime_vs_timesteps_benchmarkm --benchmark
```

---

## 📁 Directory Structure

```
include/               # All GPU- and CPU-compatible headers
GPU/tests/             # Catch2-based GPU tests
GPU/src/               # GPU-specific benchmarking and RK kernels
CPU/tests/             # CPU-only test harness
```

---

## 📄 Input Files

Use the Python scripts to generate:
- `initial_bundles_all.csv`
- `expected_trajectories_bundle_32.csv`
- `sigma_weights.csv`

These are used in test scripts and benchmarks to validate correctness and provide initial data for propagation.

---

## 📍 Notes

- **CUDA builds** require a working NVIDIA GPU and the `nvcc` compiler.
- **OpenMP** performance may vary by hardware and thread affinity.
- **CSV input/output** is structured to match trajectory formats for both testing and benchmarking.

---

## 👤 Author

Scott Blender — Final Project for RPI's Computing at Scale (2025)
