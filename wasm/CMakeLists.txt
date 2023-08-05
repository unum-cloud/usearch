include(emscripten)

set(CMAKE_C_COMPILER emcc)
set(CMAKE_CXX_COMPILER em++)

set(USEARCH_PUNNED_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/../c"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../fp16/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../robin-map/include"
)

add_executable(index lib.cpp)
target_include_directories(index PRIVATE ${USEARCH_PUNNED_INCLUDE_DIRS})
set_target_properties(index PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/wasm)
set_target_properties(index PROPERTIES CXX_STANDARD 11)
