# This CMake file is heavily inspired by following `usearch` CMake:
# https://github.com/nlohmann/json/blob/develop/CMakeLists.txt
cmake_minimum_required(VERSION 3.1)
project(usearch VERSION 0.6.0 LANGUAGES C CXX)

# Determine if USearch is built as a subproject (using `add_subdirectory`) or if it is the main project
set(USEARCH_IS_MAIN_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(USEARCH_IS_MAIN_PROJECT ON)
endif()

# Options
option(USEARCH_INSTALL "Install CMake targets" OFF)
option(USEARCH_BUILD_TEST "Compile a native unit test in C++" ${USEARCH_IS_MAIN_PROJECT})
option(USEARCH_BUILD_BENCHMARK "Compile a native benchmark in C++" ${USEARCH_IS_MAIN_PROJECT})
option(USEARCH_BUILD_WOLFRAM "Compile Wolfram Language bindings" OFF)
option(USEARCH_BUILD_CLIB "Compile C bindings" OFF)

# Includes
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(ExternalProject)

# Allow CMake 3.13+ to override options when using FetchContent / add_subdirectory
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

# Configuration
include(GNUInstallDirs)
set(USEARCH_TARGET_NAME               ${PROJECT_NAME})
set(USEARCH_CONFIG_INSTALL_DIR        "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}" CACHE INTERNAL "")
set(USEARCH_INCLUDE_INSTALL_DIR       "${CMAKE_INSTALL_INCLUDEDIR}")
set(USEARCH_TARGETS_EXPORT_NAME       "${PROJECT_NAME}Targets")
set(USEARCH_CMAKE_CONFIG_TEMPLATE     "cmake/config.cmake.in")
set(USEARCH_CMAKE_CONFIG_DIR          "${CMAKE_CURRENT_BINARY_DIR}")
set(USEARCH_CMAKE_VERSION_CONFIG_FILE "${USEARCH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(USEARCH_CMAKE_PROJECT_CONFIG_FILE "${USEARCH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Config.cmake")
set(USEARCH_CMAKE_PROJECT_TARGETS_FILE "${USEARCH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Targets.cmake")
set(USEARCH_PKGCONFIG_INSTALL_DIR      "${CMAKE_INSTALL_DATADIR}/pkgconfig")


# Define our header-only library
add_library(${USEARCH_TARGET_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${USEARCH_TARGET_NAME} ALIAS ${USEARCH_TARGET_NAME})
set(USEARCH_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include/")
if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
    target_compile_features(${USEARCH_TARGET_NAME} INTERFACE cxx_range_for)
else()
    target_compile_features(${USEARCH_TARGET_NAME} INTERFACE cxx_std_11)
endif()

target_compile_definitions(
    ${USEARCH_TARGET_NAME}
    INTERFACE
    $<$<NOT:$<BOOL:${JSON_GlobalUDLs}>>:USEARCH_USE_OPENMP=0>
)
target_include_directories(
    ${USEARCH_TARGET_NAME}
    ${USEARCH_SYSTEM_INCLUDE} INTERFACE
    $<BUILD_INTERFACE:${USEARCH_INCLUDE_BUILD_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Install a pkg-config file, so other tools can find this
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkg-config.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
)

# Install a custom package version config file instead of
# write_basic_package_version_file to ensure that it's architecture-independent
# https://github.com/nlohmann/json/issues/1697
include(CMakePackageConfigHelpers)
configure_file(
    "cmake/usearchConfigVersion.cmake.in"
    ${USEARCH_CMAKE_VERSION_CONFIG_FILE}
    @ONLY
)
configure_file(
    ${USEARCH_CMAKE_CONFIG_TEMPLATE}
    ${USEARCH_CMAKE_PROJECT_CONFIG_FILE}
    @ONLY
)

if(USEARCH_INSTALL)
    install(
        DIRECTORY ${USEARCH_INCLUDE_BUILD_DIR}
        DESTINATION ${USEARCH_INCLUDE_INSTALL_DIR}
    )
    install(
        FILES ${USEARCH_CMAKE_PROJECT_CONFIG_FILE} ${USEARCH_CMAKE_VERSION_CONFIG_FILE}
        DESTINATION ${USEARCH_CONFIG_INSTALL_DIR}
    )
    export(
        TARGETS ${USEARCH_TARGET_NAME}
        NAMESPACE ${PROJECT_NAME}::
        FILE ${USEARCH_CMAKE_PROJECT_TARGETS_FILE}
    )
    install(
        TARGETS ${USEARCH_TARGET_NAME}
        EXPORT ${USEARCH_TARGETS_EXPORT_NAME}
        INCLUDES DESTINATION ${USEARCH_INCLUDE_INSTALL_DIR}
    )
    install(
        EXPORT ${USEARCH_TARGETS_EXPORT_NAME}
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${USEARCH_CONFIG_INSTALL_DIR}
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION ${USEARCH_PKGCONFIG_INSTALL_DIR}
    )
endif()

if(${USEARCH_BUILD_TEST} OR ${USEARCH_BUILD_BENCHMARK})
    add_subdirectory(cpp)
endif()

if(${USEARCH_BUILD_WOLFRAM})
    add_subdirectory(wolfram)
endif()

if(${USEARCH_BUILD_WASM})
    add_subdirectory(wasm)
endif()

if (${USEARCH_BUILD_CLIB})
    add_subdirectory(c)
endif()
