cmake_minimum_required(VERSION 3.12)

project(usearchWFM)

set(CMAKE_CXX_STANDARD 17)

include_directories("../src")

if(NOT WOLFRAM_PATH)
    set(WOLFRAM_PATH "/home/ishkhan/wolfram/")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${WOLFRAM_PATH}/SystemFiles/Links/WSTP/DeveloperKit/Linux-x86-64/CompilerAdditions/")
include(WSTP)
include_directories("${WOLFRAM_PATH}/SystemFiles/IncludeFiles/C/")
link_directories("${WOLFRAM_PATH}/SystemFiles/Libraries/Linux-x86-64/")

set(SOURCES lib.cpp)

set(LIBRARY_NAME usearchWFM)

add_library(${LIBRARY_NAME} SHARED ${SOURCES})

# Link against the Wolfram LibraryLink library
target_link_libraries(${LIBRARY_NAME} WSTP::STATIC_LIBRARY)

# Set the output name of the library to be compatible with Wolfram Language
set_target_properties(${LIBRARY_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "usearchWFM"
)
