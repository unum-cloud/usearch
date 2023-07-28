
if(NOT WOLFRAM_PATH)
    set(WOLFRAM_PATH "/usr/local/Wolfram/Mathematica")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${WOLFRAM_PATH}/SystemFiles/Links/WSTP/DeveloperKit/Linux-x86-64/CompilerAdditions/")
include(WSTP)
include_directories("${WOLFRAM_PATH}/SystemFiles/IncludeFiles/C/")
link_directories("${WOLFRAM_PATH}/SystemFiles/Libraries/Linux-x86-64/")

add_library(usearchWFM SHARED lib.cpp)

# Link against the Wolfram LibraryLink library
target_link_libraries(usearchWFM WSTP::STATIC_LIBRARY)

# Set the output name of the library to be compatible with Wolfram Language
set_target_properties(usearchWFM PROPERTIES PREFIX "" OUTPUT_NAME "usearchWFM")
set_target_properties(usearchWFM PROPERTIES CXX_STANDARD 11)
