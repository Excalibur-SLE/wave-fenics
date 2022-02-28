cmake_minimum_required(VERSION 3.16)

project(${PROJECT_NAME} LANGUAGES C CXX CUDA)
set(PROJECT_NAME planar3d)
project(${PROJECT_NAME})

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "-Ofast ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast ${CMAKE_C_FLAGS} -Wall")

find_package(DOLFINX REQUIRED)
find_package(CUDA 11.0 REQUIRED)


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../common" ".")

# Executable
add_executable(${PROJECT_NAME} main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC SYSTEM ${CUDA_INCLUDE_DIRS})

message(STATUS ${CMAKE_CXX_FLAGS})