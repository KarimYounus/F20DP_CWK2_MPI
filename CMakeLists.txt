cmake_minimum_required(VERSION 3.17.3)
project(F20DP_CWK2_MPI C)
set(CMAKE_C_COMPILER "gcc")
#set(CMAKE_VERBOSE_MAKEFILE on)

#add_library(F20DP_CWK1_OMP STATIC SumTotient.c)
# Find the MPI package
find_package(MPI REQUIRED)
# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})
add_executable(program SumTotient.c)

# ensure the MPI libraries are linked
target_link_libraries(program ${MPI_C_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(program PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(program PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

message(STATUS "MPI_EXECUTABLE: ${MPIEXEC_EXECUTABLE}")
message(STATUS "MPI_COMPILE_FLAGS: ${MPI_COMPILE_FLAGS}")
message(STATUS "MPI_LINK_FLAGS: ${MPI_LINK_FLAGS}")
message(STATUS "MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")
#set(CMAKE_BUILD_TYPE Debug)

# Custom target to run the program
add_custom_target(run_seq
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./program ${MPIEXEC_POSTFLAGS} 0 10000 --seq
        DEPENDS program
        WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

add_custom_target(run_para
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS} ./program ${MPIEXEC_POSTFLAGS} 0 10000 --dynamic 1 --filename test.csv
        DEPENDS program
        WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)


