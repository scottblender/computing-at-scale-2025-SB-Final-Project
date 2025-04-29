#ifndef SAMPLE_CONTROLS_HOST_HPP
#define SAMPLE_CONTROLS_HOST_HPP

#include <Kokkos_Core.hpp>
#include <iostream>

#ifdef KOKKOS_ENABLE_CUDA
#include <curand_kernel.h>

// CUDA kernel for generating random samples
__global__ void generate_random_samples_kernel(
    int total_samples, 
    double* random_controls_out
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total_samples) {
        curandState state;
        curand_init(42, idx, 0, &state);  // Initialize with a fixed seed
        for (int j = 0; j < 7; ++j) {
            random_controls_out[idx * 7 + j] = curand_normal(&state);  // Generate normal random value
        }
    }
}

#else
#include <random>
#endif

// Host function to fill the HostMatrix with random samples
void sample_controls_host_host(
    int total_samples,
    Kokkos::View<double**, Kokkos::HostSpace>& random_controls_out
) {
    #ifdef KOKKOS_ENABLE_CUDA
    // CUDA-specific part: Generate random samples on the device
    // Allocate CUDA memory for the random samples
    double* d_random_controls;
    cudaError_t err = cudaMalloc((void**)&d_random_controls, total_samples * 7 * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    int block_size = 256;
    int num_blocks = (total_samples + block_size - 1) / block_size;
    generate_random_samples_kernel<<<num_blocks, block_size>>>(total_samples, d_random_controls);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_random_controls);
        return;
    }

    // Allocate Kokkos view for host memory (to hold data temporarily)
    Kokkos::View<double**, Kokkos::HostSpace> host_random_controls("host_random_controls", total_samples, 7);

    // Copy from device memory to host memory using cudaMemcpy
    cudaMemcpy(host_random_controls.data(), d_random_controls, total_samples * 7 * sizeof(double), cudaMemcpyDeviceToHost);

    // Now copy the data from the host view into the final output matrix (random_controls_out)
    for (int i = 0; i < total_samples; ++i) {
        for (int j = 0; j < 7; ++j) {
            random_controls_out(i, j) = host_random_controls(i, j);
        }
    }

    cudaFree(d_random_controls);  // Free the device memory

    #else
    // Serial-only part: Use std::mt19937 for random number generation
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);

    // Fill the matrix with random numbers
    for (int i = 0; i < total_samples; ++i) {
        for (int j = 0; j < 7; ++j) {
            random_controls_out(i, j) = dist(gen);  // Generate normal random value
        }
    }
    #endif
}

#endif // SAMPLE_CONTROLS_HOST_HPP
