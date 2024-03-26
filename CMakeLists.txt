cmake_minimum_required(VERSION 3.17.3)
project(F20DP_CWK2_MPI C)

add_library(F20DP_CWK1_OMP STATIC
        SumTotient.c)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
add_executable(program SumTotient.c)
#set(CMAKE_BUILD_TYPE Debug)

add_custom_target(run
        COMMAND program 0 1000
        DEPENDS program
        WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

