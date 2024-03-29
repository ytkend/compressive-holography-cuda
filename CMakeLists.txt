cmake_minimum_required(VERSION 3.18)
project(compressive_holography_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wnon-virtual-dtor -Woverloaded-virtual")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall --extended-lambda")

set(BUILD_TARGET "test_cu")
add_executable(${BUILD_TARGET})
target_sources(
    ${BUILD_TARGET}
    PRIVATE
    test.cu
    common.h
    image.h
    common.cuh
    fft.cuh
    fft.cu
    fista.cuh
    total_variation.cuh
    total_variation.cu
    compressive_holography.cuh
    compressive_holography.cu)

set_target_properties(${BUILD_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${BUILD_TARGET} cufft)

set(BUILD_TARGET "test")
add_executable(${BUILD_TARGET})
target_sources(
    ${BUILD_TARGET}
    PRIVATE
    test.cpp
    common.h
    image.h
    fft.h
    fft.cpp
    fista.h
    total_variation.h
    total_variation.cpp
    compressive_holography.h
    compressive_holography.cpp)

find_package(OpenMP)
target_link_libraries(${BUILD_TARGET} OpenMP::OpenMP_CXX)

# FFTW3 settings using find_package
find_package(FFTW3 REQUIRED)

# # if find_package causes errors, try pkg-config
# find_package(PkgConfig)
# pkg_check_modules(FFTW3 REQUIRED "fftw3f")

target_include_directories(${BUILD_TARGET} PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_directories(${BUILD_TARGET} PRIVATE ${FFTW3_LIBRARY_DIRS})
target_link_libraries(${BUILD_TARGET} "fftw3f" "fftw3f_omp")

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
    add_subdirectory(benchmarks)
endif()