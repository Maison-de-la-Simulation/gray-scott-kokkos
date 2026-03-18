#include <Kokkos_Core.hpp>
#include <array>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>

using namespace Kokkos;;

constexpr float KILL_RATE { 0.062f };
constexpr float FEED_RATE { 0.03f };
constexpr float DT { 1.0f };

constexpr float DIFFUSION_RATE_U { 0.1f };
constexpr float DIFFUSION_RATE_V { 0.05f };

template<typename Collection>
void drop( Collection & data, std::size_t nb_rows, std::size_t nb_cols, float value ) {

    const std::size_t center_row { nb_rows/2ul };
    const std::size_t center_col { nb_cols/2ul };
    for (std::size_t i = (center_row-2ul) ; i < (center_row+2ul); i++) {
        for (std::size_t j = (center_col-3ul); j < (center_col+3ul); j++) {
            data[i*nb_cols+j] = value ;
        }
    }

}

// From a sycnhronization point of view, we wait after each command group
// execution. We tried a queue in_order, but that was crashing or computing
// wrong results with many hardware. To be investigated.

void compute(
    float const * iu, float const * iv,
    float * ou, float * ov,
    std::size_t nb_rows, std::size_t nb_cols ) {
 
    Kokkos::parallel_for(
        "compute",
        Kokkos::TeamPolicy<>(nb_rows-2, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {

        const std::size_t row = team.league_rank();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nb_cols-2),
            [&](const std::size_t col) {

        float u = iu[(row+1)*nb_cols+col+1];
        float v = iv[(row+1)*nb_cols+col+1];
        float uvv = u*v*v;
        float full_u = 0.0f;
        float full_v = 0.0f;
        for(long k = 0l; k < 3l; ++k){
            for(long l = 0l; l < 3l; ++l){
                full_u += (iu[(row+k)*nb_cols+col+l] - u);
                full_v += (iv[(row+k)*nb_cols+col+l] - v);
            }
        }
        float du = DIFFUSION_RATE_U*full_u - uvv + FEED_RATE*(1.0f - u);
        float dv = DIFFUSION_RATE_V*full_v + uvv - (FEED_RATE + KILL_RATE)*v;
        ou[(row+1)*nb_cols+col+1] = u + du*DT;
        ov[(row+1)*nb_cols+col+1] = v + dv*DT;

            });

    });

    Kokkos::fence();

}

void swap(
    float * & u1, float * & v1,
    float * & u2, float * & v2 ) {
 
    // Submit command group for execution

        float * ut = u1, * vt = v1 ;
        u1 = u2 ; v1 = v2 ;
        u2 = ut ; v2 = vt ;

}

void store(
    float * u, float * v,
    std::size_t nb_rows, std::size_t nb_cols,
    std::vector<std::vector<float>> & uimages,
    std::vector<std::vector<float>> & vimages
    ) {
 
    std::cout << '.' << std::flush;

        uimages.emplace_back((nb_rows-2)*(nb_cols-2));
        for (std::size_t i = 0 ; i < (nb_rows-2) ; ++i) {
            for (std::size_t j = 0 ; j < (nb_cols-2) ; ++j) {
                uimages.back()[i*(nb_cols-2)+j] = u[(i+1)*nb_cols+j+1];
            }
        }

        vimages.emplace_back((nb_rows-2)*(nb_cols-2));
        for (std::size_t i = 0 ; i < (nb_rows-2) ; ++i) {
            for (std::size_t j = 0 ; j < (nb_cols-2) ; ++j) {
                vimages.back()[i*(nb_cols-2)+j] = v[(i+1)*nb_cols+j+1];
            }
        }

}

void reduce( std::vector<float> const & u, std::vector<float> const & v ) {
 
    // Submit command group for execution

        float product = std::inner_product(u.begin(),u.end(),v.begin(),0.f);
        std::cout << "\n\nproduct " << product << std::endl;

}

int main( int argc, char * argv[] ) {

    // runtime parameters
    assert(argc==6) ;
    std::size_t device_rank {std::stoul(argv[1])} ;
    std::size_t nb_rows {std::stoul(argv[2])} ;
    std::size_t nb_cols {std::stoul(argv[3])} ;
    std::size_t nb_imgs {std::stoul(argv[4])} ;
    std::size_t nb_iters {std::stoul(argv[5])} ;
    std::cout << std::fixed << std::setprecision(6);
    assert(nb_imgs<=1000);

    try {

        Kokkos::initialize(argc, argv);

        // Temporary work images
        nb_rows += 2 ;
        nb_cols += 2 ;
        float * u1  = (float*)Kokkos::kokkos_malloc(nb_rows*nb_cols*sizeof(float));
        float * u2  = (float*)Kokkos::kokkos_malloc(nb_rows*nb_cols*sizeof(float));
        float * v1  = (float*)Kokkos::kokkos_malloc(nb_rows*nb_cols*sizeof(float));
        float * v2  = (float*)Kokkos::kokkos_malloc(nb_rows*nb_cols*sizeof(float));
        for (int i = 0; i < nb_rows; i++) {
            for (int j = 0; j < nb_cols; j++) {
                u1[i*nb_cols+j] = 1.f;
                v1[i*nb_cols+j] = 0.f;
                u2[i*nb_cols+j] = 1.f;
                v2[i*nb_cols+j] = 0.f;
            }
        }
        drop(u1, nb_rows, nb_cols, 0.f);
        drop(v1, nb_rows, nb_cols, 1.f);

        // Final images sequence
        std::vector<std::vector<float>> uimages, vimages;
        uimages.emplace_back(u1,u1+nb_rows*nb_cols);
        vimages.emplace_back(v1,v1+nb_rows*nb_cols);

        // iterations
        std::cout<<std::endl ;
        for ( std::size_t image = 0 ; image < nb_imgs ; ++image ) {
            for ( std::size_t iter = 0 ; iter < nb_iters ; ++iter ) {
                compute(u1, v1, u2, v2, nb_rows, nb_cols );
                swap(u1, v1, u2, v2);
            }
            store(u1, v1, nb_rows, nb_cols, uimages, vimages);
        }
        reduce(uimages.back(), vimages.back());
            
        // Release device arrays
        Kokkos::kokkos_free(u1);
        Kokkos::kokkos_free(u2);
        Kokkos::kokkos_free(v1);
        Kokkos::kokkos_free(v2);

        Kokkos::finalize();
    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
    }
    catch (const char * e) {
      std::cout << e << std::endl;
    }

    return 0;
}