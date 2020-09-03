cmake_minimum_required(VERSION 3.5)

project(homework3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(fft fft.cpp)
target_compile_options(fft PRIVATE -fopenmp)
target_link_options(fft PRIVATE -lgomp)

add_executable(calculateBessel calculateBessel.cpp)
add_executable(calculatePi calculatePi.cpp)
