cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME bp1)
project(${PROJECT_NAME})

find_package(DOLFINX REQUIRED)
find_package(Boost 1.70 REQUIRED program_options)
find_package(CUDA 11.0 REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -Wall")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.ufl -o ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${PROJECT_NAME}.ufl
)

add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.c)

include_directories("../../../src")
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} dolfinx Boost::program_options)
target_include_directories(${PROJECT_NAME} PUBLIC SYSTEM ${CUDA_INCLUDE_DIRS})

