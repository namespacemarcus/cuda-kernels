cmake_minimum_required(VERSION 3.20.0)
project(cuda_kernels LANGUAGES CUDA CXX C)
find_package(CUDAToolkit)
add_subdirectory(kernels)