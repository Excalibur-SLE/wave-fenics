cmake_minimum_required(VERSION 3.16)

project(${PROJECT_NAME} LANGUAGES C CXX CUDA)
set(PROJECT_NAME planar3d)
project(${PROJECT_NAME})

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(DOLFINX REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(Boost 1.70 REQUIRED program_options)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../common" ".")

# Executable
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../common/cuda)
add_executable(${PROJECT_NAME} main.cpp ${SOURCE_DIR}/scatter.cu)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx CUDA::cudart CUDA::cublas Boost::program_options nvToolsExt)
target_include_directories(${PROJECT_NAME} PUBLIC SYSTEM ${CUDA_INCLUDE_DIRS})

# Custom compilation flags
#target_compile_options(${PROJECT_NAME} PUBLIC -Ofast -Wall)

message(STATUS ${CMAKE_CXX_FLAGS})
