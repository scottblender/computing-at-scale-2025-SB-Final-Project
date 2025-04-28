#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "../include/sigma_propagation_gpu.hpp"
#include "../include/runtime_propagation_test.hpp"  

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int result = 0;

    bool run_benchmark = false;

    // Check for special flag --benchmark
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--benchmark") {
            run_benchmark = true;
            break;
        }
    }

    if (run_benchmark) {
        // ==== Run the benchmark ====
        std::string backend;

        #if defined(KOKKOS_ENABLE_CUDA)
            backend = "CUDA";
        #elif defined(KOKKOS_ENABLE_OPENMP)
            backend = "OPENMP";
        #elif defined(KOKKOS_ENABLE_SERIAL)
            backend = "SERIAL";
        #else
            backend = "UNKNOWN";
        #endif

        if (backend == "UNKNOWN") {
            std::cerr << "[Warning] Unknown Kokkos backend. Proceeding anyway.\n";
        }

        std::cout << "Running benchmark for backend: " << backend << std::endl;

        PropagationSettings settings;
        settings.mu = 27.899633640439433;
        settings.F = 0.33;
        settings.c = 4.4246246663455135;
        settings.m0 = 4000.0;
        settings.g0 = 9.81;
        settings.num_subintervals = 10;
        settings.num_eval_per_step = 200;  // Always 200 RK45 steps per interval

        // Set up the refinements you want to test
        std::vector<int> refinements = {2, 3, 5, 10, 20, 50, 100,250,500,1000};

        // Output CSV file
        std::ofstream output("runtime_vs_timesteps_" + backend + ".csv");
        if (!output.is_open()) {
            std::cerr << "[ERROR] Failed to open output CSV file!\n";
            Kokkos::finalize();
            return 1;
        }
        output << "timesteps,runtime\n";

        // Run each refinement
        for (int nsteps : refinements) {
            std::cout << "\n[INFO] Running benchmark with nsteps = " << nsteps << std::endl;
            double runtime = run_propagation_test(nsteps, settings);

            if (runtime < 0.0) {
                std::cerr << "[ERROR] Benchmark failed for nsteps = " << nsteps << "\n";
            } else {
                std::cout << "Timesteps: " << nsteps << " => Runtime: " << runtime << " seconds\n";
                output << nsteps << "," << runtime << "\n";
            }
        }

        output.close();
    } else {
        std::cerr << "Error: You must run with '--benchmark' to perform benchmarking.\n";
        result = 1;
    }

    Kokkos::finalize();
    return result;
}
