cmake_minimum_required(VERSION 3.5)

project(homework3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(fft fft.cpp)
target_compile_options(fft PRIVATE -fopenmp)
target_link_options(fft PRIVATE -lgomp)

add_executable(calculateBessel calculateBessel.cpp)
add_executable(calculatePi calculatePi.cpp)

find_package(Python3 COMPONENTS Development NumPy)
target_include_directories(fft PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(fft Python3::Python Python3::NumPy)
target_include_directories(calculateBessel PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(calculateBessel Python3::Python Python3::NumPy)
target_include_directories(calculatePi PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(calculatePi  Python3::Python Python3::NumPy)
