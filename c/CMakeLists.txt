cmake_minimum_required(VERSION 3.1)
project(usearch_c C CXX)
# this already exists in the root CMakeLists.txt
# it is duplicated here to make sure users do not get a warning if
# they directly include usearch/c/CMakeLists.txt instead of the top level usearch/CMakelists.txt
if (POLICY CMP0077)
    # Allow CMake 3.13+ to override options when using FetchContent / add_subdirectory.
    cmake_policy(SET CMP0077 NEW)
endif ()

option(USEARCH_USE_SIMSIMD "Use SimSIMD hardware-accelerated metrics" OFF)
option(USEARCH_LOOKUP_LABEL "Support containment and lookup queries over labels" ON)
option(USEARCH_NO_MARCH_NATIVE "Skip -march-native flag in compilation" OFF)
option(USEARCH_BUILD_STATIC "Build a static libusearch_c library (builds shared, when OFF)" ON)
option(USEARCH_BUILD_TEST "Build a test executable" ON)
option(USEARCH_DEBUG_BUILD_ASAN "Build with address sanitizer" ON)

# Make "Release" by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# List of all possible compiler IDs:
# https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    if(${USEARCH_DEBUG_BUILD_ASAN})
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wno-conversion -Wno-unknown-pragmas")
    if (NOT ${USEARCH_NO_MARCH_NATIVE})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    if(${USEARCH_DEBUG_BUILD_ASAN})
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wfatal-errors")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /O2 /DEBUG")
endif()

set(USEARCH_PUNNED_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../fp16/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../robin-map/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../simsimd/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/"
)

# This article discusses a better way to allow building either static or shared libraries
# https://alexreinking.com/blog/building-a-dual-shared-and-static-library-with-cmake.html
if (${USEARCH_BUILD_STATIC})
    set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -static-libstdc++")
    add_library(usearch_c STATIC lib.cpp)
else ()
    add_library(usearch_c SHARED lib.cpp)
endif()

set_target_properties(usearch_c PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(usearch_c PRIVATE ${USEARCH_PUNNED_INCLUDE_DIRS})
set_target_properties(usearch_c PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(usearch_c PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(usearch_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


target_include_directories(usearch_c PRIVATE ${USEARCH_PUNNED_INCLUDE_DIRS})

if(${USEARCH_DEBUG_BUILD_ASAN})
    # for asan in debug builds
    target_link_libraries(usearch_c PRIVATE gcov)
endif()

if(${USEARCH_USE_SIMSIMD})
    target_compile_definitions(usearch_c PRIVATE USEARCH_USE_SIMSIMD=1)
endif()
if (${USEARCH_LOOKUP_LABEL})
    target_compile_definitions(usearch_c PRIVATE USEARCH_LOOKUP_LABEL=1)
endif()

if(${USEARCH_BUILD_TEST})
    add_executable(test_c test.c external_retreival_testutils.c)
    target_link_libraries(test_c usearch_c)
    set_target_properties(test_c PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

endif()