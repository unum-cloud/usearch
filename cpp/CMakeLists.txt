cmake_minimum_required(VERSION 3.1)

option(USEARCH_USE_OPENMP "Use OpenMP for a thread pool" OFF)
option(USEARCH_USE_SIMSIMD "Use SimSIMD hardware-accelerated metrics" OFF)
option(USEARCH_USE_JEMALLOC "Use JeMalloc for faster memory allocations" OFF)

# Set default build type to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Find required packages
if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif ()
find_package(Threads REQUIRED)

if (USEARCH_USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif ()

if (USEARCH_USE_JEMALLOC)
    include(jemalloc)
endif ()

# Include directories
set(USEARCH_PUNNED_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_CURRENT_SOURCE_DIR}/../fp16/include"
                                "${CMAKE_CURRENT_SOURCE_DIR}/../simsimd/include" "${CMAKE_CURRENT_SOURCE_DIR}/"
)

# Function to setup target
function (setup_target TARGET_NAME)
    # Compiler-specific options List of all possible compiler IDs:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            ${TARGET_NAME}
            PRIVATE $<$<CONFIG:RELEASE>:-O3>
                    $<$<CONFIG:DEBUG>:-g
                    $<$<CONFIG:RELWITHDEBINFO>:
                    -O3
                    -g>
                    -ffast-math
                    -fsanitize=address
                    -fsanitize=leak
                    -fsanitize=alignment
                    -fsanitize=undefined
                    -Wall
                    -Wextra
                    -Wno-conversion
                    -Wno-unknown-pragmas>
                    -march=native
                    -fmax-errors=1
                    -pedantic
                    -fdiagnostics-color=always
                    -ftime-report
        )

        if (USEARCH_USE_OPENMP)
            target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
        endif ()

        if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
            target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)
        endif ()

    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(
            ${TARGET_NAME}
            PRIVATE $<$<CONFIG:RELEASE>:-O3>
                    $<$<CONFIG:DEBUG>:-g
                    $<$<CONFIG:RELWITHDEBINFO>:-O3
                    -ffast-math
                    -fsanitize=address
                    -fsanitize=leak
                    -fsanitize=alignment
                    -fsanitize=undefined
                    -Wfatal-errors>
                    -g>
                    -pedantic
                    -fcolor-diagnostics
                    -ftime-trace
        )

        if (USEARCH_USE_OPENMP)
            target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
        endif ()

        if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
            target_link_libraries(
                ${TARGET_NAME} PRIVATE "-framework CoreFoundation" "-framework Security" Threads::Threads
            )
        else ()
            target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)
        endif ()

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "NVIDIA" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        target_compile_options(${TARGET_NAME} PRIVATE $<$<CONFIG:RELEASE>:-O3>)
        target_compile_definitions(${TARGET_NAME} PRIVATE --expt-relaxed-constexpr --extended-lambda)
        set_property(SOURCE bench.cpp PROPERTY LANGUAGE CUDA)
        set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "86")

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(${TARGET_NAME} PRIVATE -w -ferror-limit=1)

    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(
            ${TARGET_NAME} PRIVATE $<$<CONFIG:DEBUG>:/DEBUG> $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:RELWITHDEBINFO>:/O2
                                   /DEBUG>
        )
    endif ()

    target_include_directories(${TARGET_NAME} PRIVATE ${USEARCH_PUNNED_INCLUDE_DIRS})
    set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)

endfunction ()

# Setup targets
if (USEARCH_BUILD_TEST)
    add_executable(test test.cpp)
    setup_target(test)
    include(CTest)
    enable_testing()
    add_test(NAME test COMMAND test)
endif ()

if (USEARCH_BUILD_BENCHMARK)
    include(clipp)
    add_executable(bench bench.cpp)
    setup_target(bench)
    target_include_directories(bench PRIVATE ${clipp_SOURCE_DIR}/include)
endif ()
