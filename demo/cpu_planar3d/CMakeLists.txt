cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME planar3d)
project(${PROJECT_NAME})

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/forms.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/forms.ufl -o ${CMAKE_CURRENT_SOURCE_DIR}/
  DEPENDS forms.ufl
)

find_package(DOLFINX REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../common" ".")

# Executable
add_executable(${PROJECT_NAME} forms.c main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx)

# Custom compilation flags
target_compile_options(${PROJECT_NAME} PUBLIC -Ofast -march=native -mprefer-vector-width=512 -Wall)

message(STATUS ${CMAKE_CXX_FLAGS})
