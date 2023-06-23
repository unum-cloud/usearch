
if(NOT WOLFRAM_PATH)
    set(WOLFRAM_PATH "/usr/local/Wolfram/Mathematica")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/LibraryResources/Linux-x86-64")

set(USEARCH_PUNNED_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../fp16/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../robin-map/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../simsimd/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/"
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${WOLFRAM_PATH}/SystemFiles/Links/WSTP/DeveloperKit/Linux-x86-64/CompilerAdditions/")
include(WSTP)
include_directories("${WOLFRAM_PATH}/SystemFiles/IncludeFiles/C/" ${USEARCH_PUNNED_INCLUDE_DIRS})
link_directories("${WOLFRAM_PATH}/SystemFiles/Libraries/Linux-x86-64/")

add_library(usearchWFM SHARED lib.cpp)

# Link against the Wolfram LibraryLink library
target_link_libraries(usearchWFM WSTP::STATIC_LIBRARY)

# Set the output name of the library to be compatible with Wolfram Language
set_target_properties(usearchWFM PROPERTIES PREFIX "" OUTPUT_NAME "usearchWFM")
