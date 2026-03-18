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

template<typename ActiveImage>
void extract( std::size_t nb_rows, std::size_t nb_cols, real const * extended, ActiveImage & basic ) {
    for (std::size_t row { 0l } ; row < nb_rows ; row++) {
        auto begin1 { std::begin(basic)+row*nb_cols } ;
        auto begin2 { extended+(row+1)*(nb_cols+2)+1 } ;
        auto end2 { begin2+nb_cols } ;
        std::copy(begin2,end2,begin1) ;
    }
}

void compute(
    Kokkos::View<real**,Kokkos::LayoutRight> & iub,
    Kokkos::View<real**,Kokkos::LayoutRight> & ivb,
    Kokkos::View<real**,Kokkos::LayoutRight> & oub,
    Kokkos::View<real**,Kokkos::LayoutRight> & ovb,
    std::size_t nb_rows_ext, std::size_t nb_cols_ext ) {
 
    // Define the kernel
    Kokkos::parallel_for(
        "compute",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nb_rows_ext-2,nb_cols_ext-2}),
        KOKKOS_LAMBDA(const int row, const int col) {

            real u = iub(row+1,col+1);
            real v = ivb(row+1,col+1);
            real uvv = u*v*v;

            real full_u = REAL0;
            real full_v = REAL0;
            for(std::size_t k = 0l; k < 3l; ++k){
                for(std::size_t l = 0l; l < 3l; ++l){
                    full_u += (iub(row+k,col+l) - u);
                    full_v += (ivb(row+k,col+l) - v);
                }
            }

            real du = DIFFUSION_RATE_U*full_u - uvv + FEED_RATE*(REAL1 - u);
            real dv = DIFFUSION_RATE_V*full_v + uvv - (FEED_RATE + KILL_RATE)*v;

            oub(row+1,col+1) = u + du*DT;
            ovb(row+1,col+1) = v + dv*DT;
        });

}

int main( int argc, char * argv[] ) {

    // runtime parameters
    assert(argc==6) ;
    std::string_view device_type {argv[1]} ;
    std::size_t nb_rows {std::stoul(argv[3])};
    std::size_t nb_cols {std::stoul(argv[2])};
    std::size_t nb_imgs {std::stoul(argv[4])};
    std::size_t nb_iters {std::stoul(argv[5])};
    std::cout << std::fixed << std::setprecision(6);

    try {

        Kokkos::initialize(argc, argv);

        // HDF5 : prepare output file
        H5::FileAccPropList fapl;
        fapl.setFcloseDegree(H5F_CLOSE_STRONG);  // Ensure immediate flush
        fapl.setCache(0, 0, 0, 0.0); // Optional: Set chunk cache size to 0
        H5::H5File file("/lustre/fswork/projects/idris/sos/ssos044/codes/Gray-Scott-School/GrayScottSyclBench_to_kokkos/BuffersAccessors_to_Kokkos/results/gray-scott.h5", H5F_ACC_TRUNC, H5::FileCreatPropList::DEFAULT, fapl);
        hsize_t dims_3d[3] = {nb_imgs+1, nb_rows, nb_cols};
        hsize_t dims_2d[2] = {nb_rows, nb_cols};
        H5::DataSpace space_3d(3, dims_3d);
        H5::DataSpace space_2d(2, dims_2d);
        H5::DataSet dataset = file.createDataSet("matrix", real_hdf5, space_3d );

        std::cout << std::endl;
        std::cout << "Kokkos execution space: "
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
        std::vector<real> u1(nb_rows_ext*nb_cols_ext,REAL0);
        std::vector<real> v1(nb_rows_ext*nb_cols_ext,REAL0);
        std::vector<real> u2(nb_rows_ext*nb_cols_ext,REAL0);
        std::vector<real> v2(nb_rows_ext*nb_cols_ext,REAL0);
        insert(nb_rows,nb_cols,u0,u1) ;
        insert(nb_rows,nb_cols,v0,v1) ;

        // Kind of mean V value
        real result { REAL0 } ;
        const real coef { REAL1/nb_rows*16/nb_cols*16 } ;
        result += std::reduce(v0.begin(),v0.end(),REAL0)*coef ;
        
        // HDF5 : write first image
        hsize_t start[3] {0, 0, 0};
        hsize_t count[3] {1, nb_rows, nb_cols};
        space_3d.selectHyperslab(H5S_SELECT_SET, count, start);
        dataset.write(v0.data(), real_hdf5, space_2d, space_3d);

        // Create views
        Kokkos::View<real**,Kokkos::LayoutRight> u1b("u1b",nb_rows_ext,nb_cols_ext);
        Kokkos::View<real**,Kokkos::LayoutRight> v1b("v1b",nb_rows_ext,nb_cols_ext);
        Kokkos::View<real**,Kokkos::LayoutRight> u2b("u2b",nb_rows_ext,nb_cols_ext);
        Kokkos::View<real**,Kokkos::LayoutRight> v2b("v2b",nb_rows_ext,nb_cols_ext);

        auto u1h = Kokkos::create_mirror_view(u1b);
        auto v1h = Kokkos::create_mirror_view(v1b);

        for(std::size_t i=0;i<nb_rows_ext*nb_cols_ext;i++){
            u1h.data()[i]=u1[i];
            v1h.data()[i]=v1[i];
        }

        Kokkos::deep_copy(u1b,u1h);
        Kokkos::deep_copy(v1b,v1h);

        auto * u1bp = &u1b;
        auto * v1bp = &v1b;
        auto * u2bp = &u2b;
        auto * v2bp = &v2b;
        
        // Iterations
        std::cout<<std::endl ;
        for ( std::size_t image = 0 ; image < nb_imgs ; ++image ) {
            for ( std::size_t iter = 0 ; iter < nb_iters ; ++iter ) {
                compute( *u1bp, *v1bp, *u2bp, *v2bp, nb_rows_ext, nb_cols_ext );
                std::swap(u1bp,u2bp) ;
                std::swap(v1bp,v2bp) ;
            }
            Kokkos::fence();

            auto v1h2 = Kokkos::create_mirror_view(*v1bp);
            Kokkos::deep_copy(v1h2,*v1bp);
            extract(nb_rows,nb_cols,v1h2.data(),v0) ;

            result += std::reduce(v0.begin(),v0.end(),REAL0)*coef ;

            // HDF5 : write imagemake
            hsize_t start[3] {image+1, 0, 0};
            space_3d.selectHyperslab(H5S_SELECT_SET, count,start);
            dataset.write(v0.data(), real_hdf5, space_2d, space_3d);

            // Progress point
            std::cout << '.' << std::flush;
        }
        std::cout << std::endl;
    
        // Final result
        std::cout << "Result: " << result/(nb_imgs+1) << std::endl;

        Kokkos::finalize();
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
    return 0;
}