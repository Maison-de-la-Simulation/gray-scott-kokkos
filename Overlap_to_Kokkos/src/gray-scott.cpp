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

template<typename ActiveImage, typename ExtendedImage>
void insert( std::size_t nb_rows, std::size_t nb_cols, ActiveImage const & basic, ExtendedImage & extended ) {
    for (std::size_t row { 0l } ; row < nb_rows ; row++) {
        auto begin1 { std::begin(basic)+row*nb_cols } ;
        auto end1 { begin1+nb_cols } ;
        auto begin2 { std::begin(extended)+(row+1)*(nb_cols+2)+1 } ;
        std::copy(begin1,end1,begin2) ;
    }
}

template<typename ActiveImage, typename ExtendedImage>
void extract( std::size_t nb_rows, std::size_t nb_cols, ExtendedImage const & extended, ActiveImage & basic ) {
    for (std::size_t row { 0l } ; row < nb_rows ; row++) {
        auto begin1 { std::begin(basic)+row*nb_cols } ;
        auto begin2 { std::begin(extended)+(row+1)*(nb_cols+2)+1 } ;
        auto end2 { begin2+nb_cols } ;
        std::copy(begin2,end2,begin1) ;
    }
}

void iterate(
    real * u1d, real * v1d, real * u2d, real * v2d,
    std::size_t nb_rows_ext, std::size_t nb_cols_ext,
    std::size_t nb_iters ) {

    // Iterations
    for ( std::size_t iter = 0ul ; iter < nb_iters ; iter++ ) {
        
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {0,0},
                {nb_rows_ext-2,nb_cols_ext-2}),
            KOKKOS_LAMBDA(const std::size_t row, const std::size_t col) {
                
                real u = u1d[(row+1)*nb_cols_ext+col+1];
                real v = v1d[(row+1)*nb_cols_ext+col+1];
                real uvv = u*v*v;
                
                real full_u = REAL0;
                real full_v = REAL0;
                for(long k = 0l; k < 3l; ++k){
                    for(long l = 0l; l < 3l; ++l){
                        full_u += (u1d[(row+k)*nb_cols_ext+col+l] - u);
                        full_v += (v1d[(row+k)*nb_cols_ext+col+l] - v);
                    }
                }
                
                real du = DIFFUSION_RATE_U*full_u - uvv + FEED_RATE*(REAL1 - u);
                real dv = DIFFUSION_RATE_V*full_v + uvv - (FEED_RATE + KILL_RATE)*v;
                
                u2d[(row+1)*nb_cols_ext+col+1] = u + du*DT;
                v2d[(row+1)*nb_cols_ext+col+1] = v + dv*DT;
            });

        std::swap(u1d,u2d);
        std::swap(v1d,v2d);
    }

}

void shelve(
    real * from, real * to,
    std::size_t nb_rows_ext, std::size_t nb_cols_ext ) {

    Kokkos::deep_copy(
        Kokkos::View<real*,Kokkos::DefaultExecutionSpace::memory_space>(to,nb_rows_ext*nb_cols_ext),
        Kokkos::View<real*,Kokkos::DefaultExecutionSpace::memory_space>(from,nb_rows_ext*nb_cols_ext)
    );

}

void download(
    real const * from, std::vector<real> & to,
    std::size_t nb_rows_ext, std::size_t nb_cols_ext ) {

    auto dview = Kokkos::View<const real*,Kokkos::DefaultExecutionSpace::memory_space>(from,nb_rows_ext*nb_cols_ext);
    auto hview = Kokkos::create_mirror_view(dview);
    Kokkos::deep_copy(hview,dview);
    std::copy(hview.data(),hview.data()+nb_rows_ext*nb_cols_ext,to.data());

}

int main( int argc, char * argv[] ) {

    Kokkos::initialize(argc,argv);
    {

    // runtime parameters
    assert(argc==6) ;
    std::string_view device_type {argv[1]} ;
    std::size_t nb_cols {std::stoul(argv[2])};
    std::size_t nb_rows {std::stoul(argv[3])};
    std::size_t nb_imgs {std::stoul(argv[4])};
    std::size_t nb_iters {std::stoul(argv[5])};
    std::cout << std::fixed << std::setprecision(6);
    assert(nb_rows%30==0);
    assert(nb_cols%32==0);
    assert(nb_iters%2==0);

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
        hsize_t count[3] {1, nb_rows, nb_cols};

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
        std::vector<real> vh_ext(nb_rows_ext*nb_cols_ext,REAL0);
        std::vector<real> uh_ext(nb_rows_ext*nb_cols_ext,REAL0);
        insert(nb_rows,nb_cols,u0,uh_ext) ;
        insert(nb_rows,nb_cols,v0,vh_ext) ;

        using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

        Kokkos::View<real*,memory_space> u1d("u1d",nb_rows_ext*nb_cols_ext);
        Kokkos::View<real*,memory_space> v1d("v1d",nb_rows_ext*nb_cols_ext);
        Kokkos::View<real*,memory_space> u2d("u2d",nb_rows_ext*nb_cols_ext);
        Kokkos::View<real*,memory_space> v2d("v2d",nb_rows_ext*nb_cols_ext);
        Kokkos::View<real*,memory_space> v3d("v3d",nb_rows_ext*nb_cols_ext);

        auto hu = Kokkos::create_mirror_view(u1d);
        auto hv = Kokkos::create_mirror_view(v1d);
        std::copy(uh_ext.begin(),uh_ext.end(),hu.data());
        std::copy(vh_ext.begin(),vh_ext.end(),hv.data());
        Kokkos::deep_copy(u1d,hu);
        Kokkos::deep_copy(v1d,hv);
        Kokkos::deep_copy(u2d,hu);
        Kokkos::deep_copy(v2d,hv);

        // Kind of mean V value
        real result { REAL0 } ;
        const real coef { REAL1/nb_rows*16/nb_cols*16 } ;
        
        auto analyze = [&]( std::size_t rank ){
            extract(nb_rows,nb_cols,vh_ext,v0) ;
            result += std::reduce(v0.begin(),v0.end(),REAL0)*coef ;
            hsize_t start[3] {rank, 0, 0};
            space_3d.selectHyperslab(H5S_SELECT_SET, count, start);
            dataset.write(v0.data(), real_hdf5, space_2d, space_3d);
        };
    
        std::cout<<std::endl ;
        for ( std::size_t image = 0 ; image < nb_imgs ; ++image ) {
            shelve( v1d.data(), v3d.data(), nb_rows_ext, nb_cols_ext );
            iterate( u1d.data(), v1d.data(), u2d.data(), v2d.data(), nb_rows_ext, nb_cols_ext, nb_iters );
            download( v3d.data(), vh_ext, nb_rows_ext, nb_cols_ext );
            Kokkos::fence();
            analyze( image );
            std::cout << '.' << std::flush;
        }
        std::cout << std::endl;
    
        download( v1d.data(), vh_ext, nb_rows_ext, nb_cols_ext );
        Kokkos::fence();
        analyze( nb_imgs );

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
        std::cout << "COUCOU" << std::endl;
    std::cout << std::endl;
    }
    Kokkos::finalize();
    return 0;
}