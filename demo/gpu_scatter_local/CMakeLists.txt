cmake_minimum_required(VERSION 3.16)

project(${PROJECT_NAME} LANGUAGES C CXX CUDA)
set(PROJECT_NAME scatter)
project(${PROJECT_NAME})

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(DOLFINX REQUIRED)
find_package(CUDA 11.0 REQUIRED)
find_package(Boost 1.70 REQUIRED program_options)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../common" ".")

# Executable
set(CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../common/cuda/)
add_executable(${PROJECT_NAME} main.cpp ${CUDA_SOURCES}/scatter.cu)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} Boost::program_options)
target_include_directories(${PROJECT_NAME} PUBLIC SYSTEM ${CUDA_INCLUDE_DIRS})

# Custom compilation flags
target_compile_options(${PROJECT_NAME} PUBLIC -O3)

message(STATUS ${CMAKE_CXX_FLAGS})
