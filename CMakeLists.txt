cmake_minimum_required(VERSION 3.28)

project(eigen_test)

find_package(benchmark REQUIRED CONFIG)
find_package(Eigen3 REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE Eigen3::Eigen benchmark::benchmark_main)
target_compile_definitions(main PUBLIC EIGEN_RUNTIME_NO_MALLOC)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3")
