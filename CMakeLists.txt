cmake_minimum_required(VERSION 3.28)

project(fitter)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ROOT_DIR $ENV{SIMPATH}/share/root/cmake)

message(STATUS "CMAKE VAR: ${ROOT_DIR}")

find_package(benchmark REQUIRED CONFIG)
find_package(Torch REQUIRED)
find_package(gsl-lite REQUIRED)
find_package(ROOT REQUIRED COMPONENTS MathCore MathMore)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main main.cpp)
target_link_libraries(main PRIVATE benchmark::benchmark_main ${TORCH_LIBRARIES}
                                   ROOT::MathCore ROOT::MathMore)

add_executable(test test_torch.cpp)
target_link_libraries(test PRIVATE ${TORCH_LIBRARIES} gsl-lite::gsl-lite)

add_executable(test_tensor test_tensor.cpp)
# target_link_libraries(test_tensor PRIVATE ${TORCH_LIBRARIES} gsl-lite::gsl-lite)

add_executable(test_root test_root.cpp)
target_link_libraries(test_root PRIVATE ROOT::MathCore ROOT::MathMore)
