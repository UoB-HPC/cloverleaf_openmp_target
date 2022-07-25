cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(cloverleaf_openmp_plain)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE YES)


if (MPI_AS_LIBRARY)

    if (NOT DEFINED MPI_C_LIB_DIR)
        message(FATAL_ERROR "MPI_C_LIB_DIR must be specified, typically <mpi_root_dir>/lib")
    endif ()

    if (NOT DEFINED MPI_C_INCLUDE_DIR)
        message(FATAL_ERROR "MPI_C_INCLUDE_DIR must be specified, typically <mpi_root_dir>/include")
    endif ()

    if (NOT DEFINED MPI_C_LIB)
        message(FATAL_ERROR "MPI_C_LIB must be specified, for example: mpich for libmpich.so in MPI_C_LIB_DIR")
    endif ()

    message(STATUS "Using MPI as a library (${MPI_C_LIB})")
    message(STATUS "MPI include dir: ${MPI_C_INCLUDE_DIR}")
    message(STATUS "MPI library dir: ${MPI_C_LIB_DIR}")

    include_directories(${MPI_C_INCLUDE_DIR})
    link_directories(${MPI_C_LIB_DIR})
else ()
    find_package(MPI REQUIRED)
    set(MPI_C_LIB MPI::MPI_CXX)
endif ()

set(SOURCES
        src/accelerate.cpp
        src/advec_cell.cpp
        src/advec_mom.cpp
        src/advection.cpp
        src/build_field.cpp
        src/calc_dt.cpp
        src/clover_leaf.cpp
        src/comms.cpp
        src/field_summary.cpp
        src/flux_calc.cpp
        src/generate_chunk.cpp
        src/hydro.cpp
        src/ideal_gas.cpp
        src/initialise_chunk.cpp
        src/initialise.cpp
        src/pack_kernel.cpp
        src/PdV.cpp
        src/read_input.cpp
        src/report.cpp
        src/reset_field.cpp
        src/revert.cpp
        src/start.cpp
        src/timer.cpp
        src/timestep.cpp
        src/update_halo.cpp
        src/update_tile_halo.cpp
        src/update_tile_halo_kernel.cpp
        src/viscosity.cpp
        src/visit.cpp)

include_directories(src)
add_executable(clover_leaf ${SOURCES})

target_compile_options(clover_leaf
        PUBLIC
        -Wall
        -Wextra
        -Wcast-align
#        -Wfatal-errors
        -Werror=return-type
        -Wno-unused-parameter
        -Wno-unused-variable
#        -Wno-ignored-attributes

        ${EXTRA_FLAGS}
        )

separate_arguments(CXX_EXTRA_FLAGS)
separate_arguments(CXX_EXTRA_LINKER_FLAGS)



set(DEBUG_OPTIONS -O2 -fno-omit-frame-pointer ${CXX_EXTRA_FLAGS})
set(RELEASE_OPTIONS -O3 ${CXX_EXTRA_FLAGS})

target_link_libraries(clover_leaf PUBLIC ${MPI_C_LIB})
target_link_libraries(clover_leaf PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C)


target_compile_options(clover_leaf PUBLIC "$<$<CONFIG:RelWithDebInfo>:${RELEASE_OPTIONS}>")
target_compile_options(clover_leaf PUBLIC "$<$<CONFIG:Release>:${RELEASE_OPTIONS}>")
target_compile_options(clover_leaf PUBLIC "$<$<CONFIG:Debug>:${DEBUG_OPTIONS}>")


if (${CMAKE_VERSION} VERSION_LESS "3.13.0")
    message(WARNING "target_link_options is only available in CMake >= 3.13.0, using fallback target_link_libraries, this may cause issues with some compilers")

    set(EXTRA_LINK_FLAGS ${OpenMP_CXX_FLAGS})

    if (DEFINED CXX_EXTRA_LINKER_FLAGS)
        list(APPEND EXTRA_LINK_FLAGS "-Wl,${CXX_EXTRA_LINKER_FLAGS}")
    endif ()

    target_link_libraries(clover_leaf PUBLIC ${EXTRA_LINK_FLAGS})

else ()
    target_link_options(clover_leaf PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_options(clover_leaf PUBLIC LINKER:${CXX_EXTRA_LINKER_FLAGS})
endif ()

