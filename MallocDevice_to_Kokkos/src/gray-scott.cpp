#include <Kokkos_Core.hpp>
#include <H5Cpp.h>
#include <array>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <string_view>

using namespace std::literals;

using real = float ;
constexpr real REAL0 { 0.l } ;
constexpr real REAL1 { 1.l } ;

auto real_hdf5 { H5::PredType::NATIVE_FLOAT } ;

constexpr real KILL_RATE { 0.054l };
constexpr real FEED_RATE { 0.014l };
constexpr real DT { 1.0l };
constexpr real DIFFUSION_RATE_U { 0.1l };
constexpr real DIFFUSION_RATE_V { 0.05l };

template<typename ActiveImage>
void drop( ActiveImage & data, std::size_t nb_rows, std::size_t nb_cols, real value ) {
    const std::size_t row_begin { 7*nb_rows/16-4 } ;
    const std::size_t row_end { 8*nb_rows/16-4 } ;
    const std::size_t col_begin { 7*nb_cols/16 } ;
    const std::size_t col_end { 8*nb_cols/16 } ;
    for (std::size_t row { row_begin } ; row < row_end ; row++) {
        for (std::size_t col { col_begin } ; col < col_end ; col++) {
            data[row*nb_cols+col] = value ;
        }
    }
}

template<typename ActiveImage>
void insert( std::size_t nb_rows, std::size_t nb_cols, ActiveImage const & basic,real * extended ) {
    for (std::size_t row { 0l } ; row < nb_rows ; row++) {
        auto begin1 { std::begin(basic)+row*nb_cols } ;
        auto end1 { begin1+nb_cols } ;
        auto begin2 { extended+(row+1)*(nb_cols+2)+1 } ;
        std::copy(begin1,end1,begin2) ;
    }
}

template<typename ActiveImage>
void extract( std::size_t nb_rows, std::size_t nb_cols, real const * extended, ActiveImage & basic ) {
    for (std::size_t row { 0l } ; row < nb_rows ; row++) {
        auto begin1 { std::begin(basic)+row*nb_cols } ;
        auto begin2 { extended+(row+1)*(nb_cols+2)+1 } ;
        auto end2 { begin2+nb_cols } ;
        std::copy(begin2,end2,begin1) ;
    }
}

// From a synchronization point of view, we wait after each command group
// execution. We tried a queue in_order, but that was crashing or computing
// wrong results with many hardware. To be investigated.

void compute(
    real const * iu, real const * iv,
    real * ou, real * ov,
    std::size_t nb_rows_ext, std::size_t nb_cols_ext ) {
 
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},
            {nb_rows_ext-2,nb_cols_ext-2}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t col) {

        real u = iu[(row+1)*nb_cols_ext+col+1];
        real v = iv[(row+1)*nb_cols_ext+col+1];
        real uvv = u*v*v;
        real full_u = REAL0;
        real full_v = REAL0;
        for(std::size_t k = 0l; k < 3l; ++k){
            for(std::size_t l = 0l; l < 3l; ++l){
                full_u += (iu[(row+k)*nb_cols_ext+col+l] - u);
                full_v += (iv[(row+k)*nb_cols_ext+col+l] - v);
            }
        }
        real du = DIFFUSION_RATE_U*full_u - uvv + FEED_RATE*(REAL1 - u);
        real dv = DIFFUSION_RATE_V*full_v + uvv - (FEED_RATE + KILL_RATE)*v;
        ou[(row+1)*nb_cols_ext+col+1] = u + du*DT;
        ov[(row+1)*nb_cols_ext+col+1] = v + dv*DT;

    });

    Kokkos::fence();

}

int main( int argc, char * argv[] ) {

    Kokkos::initialize(argc,argv);
    {
    // runtime parameters
    assert(argc==6) ;
    std::string_view device_type {argv[1]} ;
    std::size_t nb_rows {std::stoul(argv[2])} ;
    std::size_t nb_cols {std::stoul(argv[3])} ;
    std::size_t nb_imgs {std::stoul(argv[4])} ;
    std::size_t nb_iters {std::stoul(argv[5])} ;
    std::cout << std::fixed << std::setprecision(6);

    try {
        
        // HDF5 : prepare output file
        H5::FileAccPropList fapl;
        fapl.setFcloseDegree(H5F_CLOSE_STRONG);  // Ensure immediate flush
        fapl.setCache(0, 0, 0, 0.0); // Optional: Set chunk cache size to 0
        H5::H5File file("/dev/shm/gray-scott.h5", H5F_ACC_TRUNC, H5::FileCreatPropList::DEFAULT, fapl);
        hsize_t dims_3d[3] = {nb_imgs+1, nb_rows, nb_cols};
        hsize_t dims_2d[2] = {nb_rows, nb_cols};
        H5::DataSpace space_3d(3, dims_3d);
        H5::DataSpace space_2d(2, dims_2d);
        H5::DataSet dataset = file.createDataSet("matrix", real_hdf5, space_3d );

        std::cout << std::endl;
        std::cout << "Execution space: "
          << typeid(Kokkos::DefaultExecutionSpace).name()
          << std::endl;
        std::cout << "Iterations: "
          << nb_imgs << "*" << nb_iters
          << std::endl;

        // Basic images
        std::vector<real> u0(nb_rows*nb_cols,REAL1);
        std::vector<real> v0(nb_rows*nb_cols,REAL0);
        drop(u0, nb_rows, nb_cols, REAL0);
        drop(v0, nb_rows, nb_cols, REAL1);
        
        // Extended images (with a 0 border)
        std::size_t nb_rows_ext { nb_rows+2ul } ;
        std::size_t nb_cols_ext { nb_cols+2ul } ;

        using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

        Kokkos::View<real*,memory_space> u1("u1",nb_rows_ext*nb_cols_ext);
        Kokkos::View<real*,memory_space> u2("u2",nb_rows_ext*nb_cols_ext);
        Kokkos::View<real*,memory_space> v1("v1",nb_rows_ext*nb_cols_ext);
        Kokkos::View<real*,memory_space> v2("v2",nb_rows_ext*nb_cols_ext);

        Kokkos::deep_copy(u1,REAL0);
        Kokkos::deep_copy(u2,REAL0);
        Kokkos::deep_copy(v1,REAL0);
        Kokkos::deep_copy(v2,REAL0);

        auto hu1 = Kokkos::create_mirror_view(u1);
        auto hv1 = Kokkos::create_mirror_view(v1);

        insert(nb_rows,nb_cols,u0,hu1.data()) ;
        insert(nb_rows,nb_cols,v0,hv1.data()) ;

        Kokkos::deep_copy(u1,hu1);
        Kokkos::deep_copy(v1,hv1);
        
        // Kind of mean V value
        real result { REAL0 } ;
        const real coef { REAL1/nb_rows*16/nb_cols*16 } ;
        result += std::reduce(v0.begin(),v0.end(),REAL0)*coef ;
        
        // HDF5 : write first image
        hsize_t start[3] {0, 0, 0};
        hsize_t count[3] {1, nb_rows, nb_cols};
        space_3d.selectHyperslab(H5S_SELECT_SET, count, start);
        dataset.write(v0.data(), real_hdf5, space_2d, space_3d);

        // Iterations
        std::cout<<std::endl ;
        for ( std::size_t image = 0 ; image < nb_imgs ; ++image ) {
            for ( std::size_t iter = 0 ; iter < nb_iters ; ++iter ) {
                compute(u1.data(), v1.data(), u2.data(), v2.data(), nb_rows_ext, nb_cols_ext );
                std::swap(u1,u2) ;
                std::swap(v1,v2) ;
            }
            std::cout << '.' << std::flush;
            // Upgrade pseudo-result
            auto hv = Kokkos::create_mirror_view(v1);
            Kokkos::deep_copy(hv,v1);
            extract(nb_rows,nb_cols,hv.data(),v0) ;
            result += std::reduce(v0.begin(),v0.end(),REAL0)*coef ;
            // HDF5 : write image
            hsize_t start[3] {image+1, 0, 0};
            space_3d.selectHyperslab(H5S_SELECT_SET, count,start);
            dataset.write(v0.data(), real_hdf5, space_2d, space_3d);
        }
        std::cout << std::endl;
    
        // Final result
        std::cout << "Result: " << result/(nb_imgs+1) << std::endl;
            
    }
    catch (H5::FileIException & error) {
        error.printErrorStack();
        return -1;
    }
    catch (H5::DataSetIException & error) {
        error.printErrorStack();
        return -1;
    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
    }
    catch (const char * e) {
      std::cout << e << std::endl;
    }

    // End
    std::cout << std::endl;
    }
    Kokkos::finalize();
    return 0;
}