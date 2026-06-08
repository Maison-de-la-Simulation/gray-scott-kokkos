#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    Kokkos::printf("Hello Gray-Scott School!\n");
}
