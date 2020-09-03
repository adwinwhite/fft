cmake_minimum_required(VERSION 3.5)

project(homework3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(homework3 main.cpp)
target_compile_options(homework3 PRIVATE -fopenmp)
target_link_options(homework3 PRIVATE -lgomp)

