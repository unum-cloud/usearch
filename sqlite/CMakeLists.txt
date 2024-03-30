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
    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
        FetchContent_Declare(
            sqlite3 URL https://sqlite.org/2024/sqlite-amalgamation-3450200.zip DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
    else ()
        FetchContent_Declare(sqlite3 URL https://sqlite.org/2024/sqlite-amalgamation-3450200.zip)
    endif ()
    FetchContent_MakeAvailable(sqlite3)
    add_library(sqlite3_shared SHARED ${sqlite3_SOURCE_DIR}/sqlite3.c)
    target_include_directories(sqlite3_shared PUBLIC ${sqlite3_SOURCE_DIR})
    set_target_properties(sqlite3_shared PROPERTIES OUTPUT_NAME "sqlite3")
    set(SQLite3_INCLUDE_DIR ${sqlite3_SOURCE_DIR})
    set(SQLite3_LIBRARY $<TARGET_FILE:sqlite3_shared>)
    target_link_libraries(usearch_sqlite PRIVATE sqlite3_shared)

    target_include_directories(usearch_sqlite PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../stringzilla/include)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(usearch_sqlite PRIVATE -Wno-vla -Wno-unused-function -Wno-cast-function-type)
    endif ()
    setup_target(usearch_sqlite)

endif ()
