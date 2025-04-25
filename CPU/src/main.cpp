#include <catch2/catch_session.hpp>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int result = Catch::Session().run(argc, argv);
    Kokkos::finalize();
    return result;
}
