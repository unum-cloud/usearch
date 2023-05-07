cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(usearch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

option(USEARCH_USE_JEMALLOC "Use JeMalloc for Faster Memory Allocations" OFF)
option(USEARCH_USE_OPENMP "Use OpenMP for a thread pool" OFF)
option(USEARCH_USE_SIMD "Use SIMD hardware acceleration" OFF)

option(USEARCH_BUILD_BENCHMARK "Compile a native benchmark in C++" ON)
option(USEARCH_BUILD_WOLFRAM "Compile Wolfram Language bindings" OFF)

# Make "Release" by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -pg -g")

# List of all possible compiler IDs:
# https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-conversion -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVIDIA" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
    set_property(SOURCE src/bench.cpp PROPERTY LANGUAGE CUDA)
    set_target_properties(bench PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(bench PROPERTIES CUDA_ARCHITECTURES "86")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1")
endif()

if(APPLE)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

find_package(Threads REQUIRED)

if(${USEARCH_USE_OPENMP})
    find_package(OpenMP REQUIRED)
    include_directories(${OPENMP_INCLUDE_DIRS})
endif()

if(${USEARCH_USE_JEMALLOC})
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/jemalloc.cmake")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/clipp.cmake")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/fp16/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/simsimd/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

# Define our header-only library.
add_library(usearch INTERFACE)
target_sources(usearch INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/usearch/usearch.hpp")
target_include_directories(usearch INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Define the targets
if(${USEARCH_BUILD_BENCHMARK})
    add_executable(bench src/bench.cpp)
    target_link_libraries(bench PRIVATE Threads::Threads)

    if(${USEARCH_USE_SIMD})
        target_compile_definitions(bench PRIVATE USEARCH_USE_SIMD=1)
    endif()

    if(${USEARCH_USE_OPENMP})
        target_link_libraries(bench PRIVATE ${OPENMP_LIBRARIES})
    endif()

    if(${USEARCH_USE_JEMALLOC})
        target_link_libraries(bench PRIVATE ${JEMALLOC_LIBRARIES})
    endif()
endif()

if(${USEARCH_BUILD_WOLFRAM})
    add_subdirectory(wolfram)
endif()