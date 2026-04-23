cmake_minimum_required(VERSION 3.28)

project(eigen_test)

find_package(benchmark REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE benchmark::benchmark_main)
