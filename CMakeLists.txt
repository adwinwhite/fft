cmake_minimum_required(VERSION 3.5)

project(homework3 LANGUAGES CXX)

set(CMAKE_CXX_COMPILER "mpic++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fft_view_data fft_view_data.cpp)
add_executable(fft_generate_data fft_generate_data.cpp)
add_executable(verify_fft_result fft_uniprocessor.cpp verify_fft_result.cpp)

add_executable(fft_mpi fft_uniprocessor.cpp fft_mpi.cpp)
target_compile_options(fft_mpi PRIVATE -O3 -fopenmp -ffast-math -march=native -mtune=native)
target_link_options(fft_mpi PRIVATE -lgomp -lfftw3_omp -lfftw3 -lm)

#target_compile_options(verify_fft_result PRIVATE -O3 -fopenmp -ffast-math -march=native -mtune=native)
#target_link_options(verify_fft_result PRIVATE -lgomp -lfftw3_omp -lfftw3 -lm)


#add_executable(fft fft.cpp)
#target_compile_options(fft PRIVATE -O3 -fopenmp -ffast-math -march=native -mtune=native)
#target_link_options(fft PRIVATE -lgomp -lfftw3_omp -lfftw3 -lm)



#add_executable(calculateBessel calculateBessel.cpp)
#add_executable(calculatePi calculatePi.cpp)
#target_compile_options(calculatePi PRIVATE -O3 -fopenmp  -march=native -mtune=native)
#target_link_options(calculatePi PRIVATE -lgomp)

#find_package(Python3 COMPONENTS Development NumPy)
#target_include_directories(fft PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
#target_link_libraries(fft Python3::Python Python3::NumPy)
#target_include_directories(calculateBessel PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
#target_link_libraries(calculateBessel Python3::Python Python3::NumPy)
#target_include_directories(calculatePi PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
#target_link_libraries(calculatePi  Python3::Python Python3::NumPy)
