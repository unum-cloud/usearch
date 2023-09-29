if (USEARCH_BUILD_CLIB)
    add_executable(test_c test.c lib.cpp)
    setup_target(test_c)
    include(CTest)
    enable_testing()
    add_test(NAME test_c COMMAND test_c)
endif ()

# This article discusses a better way to allow building either static or shared libraries:
# https://alexreinking.com/blog/building-a-dual-shared-and-static-library-with-cmake.html
if (USEARCH_BUILD_CTEST)
    add_library(usearch_c_static STATIC lib.cpp)
    add_library(usearch_c_shared SHARED lib.cpp)

    # Set the compile options for the static library
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(usearch_c_static PRIVATE -static-libstdc++)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(usearch_c_static PRIVATE -static)
    endif ()

    setup_target(usearch_c_static)
    setup_target(usearch_c_shared)

endif ()
