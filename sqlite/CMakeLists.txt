# This article discusses a better way to allow building either static or shared libraries:
# https://alexreinking.com/blog/building-a-dual-shared-and-static-library-with-cmake.html
if (USEARCH_BUILD_SQLITE)
    find_package(SQLite3 REQUIRED)

    add_library(usearch_sqlite SHARED lib.cpp)

    # Set the compile options for the static library
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(usearch_sqlite PRIVATE -static-libstdc++)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(usearch_sqlite PRIVATE -static)
    endif ()

    target_link_libraries(usearch_sqlite PRIVATE SQLite::SQLite3)
    target_include_directories(usearch_sqlite PRIVATE ${SQLite3_INCLUDE_DIRS})

    target_include_directories(usearch_sqlite PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../stringzilla/include)
    target_compile_options(usearch_sqlite PRIVATE -fPIC)
    target_compile_options(usearch_sqlite PRIVATE -Wno-vla -Wno-unused-function -Wno-cast-function-type)
    setup_target(usearch_sqlite)

endif ()
