# This article discusses a better way to allow building either static or shared libraries:
# https://alexreinking.com/blog/building-a-dual-shared-and-static-library-with-cmake.html
if (USEARCH_BUILD_SQLITE)

    add_library(usearch_sqlite SHARED lib.cpp)

    # One option is to use the FindSQLite3.cmake module and locate the pre-installed version of SQLite, that however
    # breaks cross-compilation, so we use the FetchContent module to get the "amalgamation" version of SQLite.
    # https://sqlite.org/amalgamation.html
    #
    # find_package(SQLite3 REQUIRED) target_link_libraries(usearch_sqlite PRIVATE SQLite::SQLite3)
    # target_include_directories(usearch_sqlite PRIVATE ${SQLite3_INCLUDE_DIRS})
    include(FetchContent)
    FetchContent_Declare(sqlite3 URL https://sqlite.org/2024/sqlite-amalgamation-3450200.zip)
    FetchContent_MakeAvailable(sqlite3)
    target_link_libraries(usearch_sqlite PRIVATE sqlite3)

    target_include_directories(usearch_sqlite PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../stringzilla/include)
    target_compile_options(usearch_sqlite PRIVATE -fPIC)
    target_compile_options(usearch_sqlite PRIVATE -Wno-vla -Wno-unused-function -Wno-cast-function-type)
    setup_target(usearch_sqlite)

endif ()
