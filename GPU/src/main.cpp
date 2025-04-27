#include <catch2/catch_session.hpp>
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

    // Check if special flag "--benchmark" was passed
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--benchmark") {
            run_benchmark = true;
            break;
        }
    }

    if (run_benchmark) {
        // ======= Run your runtime vs. timestep benchmark =======
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
        settings.num_eval_per_step = 200;

        std::vector<int> refinements = {1, 2, 5, 10, 20, 50, 100};
        std::ofstream output("runtime_vs_timesteps_" + backend + ".csv");
        output << "timesteps,runtime\n";

        for (int nsteps : refinements) {
            double runtime = run_propagation_test(nsteps, settings);
            std::cout << "Timesteps: " << nsteps << " => Runtime: " << runtime << " seconds\n";
            output << nsteps << "," << runtime << "\n";
        }

        output.close();
    } else {
        // ======= Otherwise run Catch2 tests =======
        result = Catch::Session().run(argc, argv);
    }

    Kokkos::finalize();
    return result;
}
