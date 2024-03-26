cmake_minimum_required(VERSION 3.17.3)
project(F20DP_CWK2_MPI C)
set(CMAKE_C_COMPILER "gcc")

# Find the MPI package
find_package(MPI REQUIRED)

# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})

add_library(F20DP_CWK1_OMP STATIC SumTotient.c)
add_executable(program SumTotient.c)

# ensure the MPI libraries are linked
target_link_libraries(program ${MPI_C_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(program PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(program PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

#set(CMAKE_BUILD_TYPE Debug)

# Custom target to run the program
add_custom_target(run
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./program ${MPIEXEC_POSTFLAGS} 0 1000
        DEPENDS program
        WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

