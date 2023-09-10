if(USEARCH_USE_EMSCRIPTEN)
    include(emscripten)
    set(CMAKE_C_COMPILER emcc)
    set(CMAKE_CXX_COMPILER em++)
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=wasm32-wasi-threads -nostdlib -fno-exceptions -nostartfiles -Wl,--no-entry")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --target=wasm32-wasi-threads -nostdlib -fno-exceptions -nostartfiles -Wl,--no-entry")
endif()


set(USEARCH_PUNNED_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../c"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../fp16/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../robin-map/include"
)

add_executable(usearch_wasm usearch_wasm.cpp usearch-wasm.c)
target_include_directories(usearch_wasm PRIVATE ${USEARCH_PUNNED_INCLUDE_DIRS})
if(NOT USEARCH_USE_EMSCRIPTEN)
    target_compile_definitions(usearch_wasm PRIVATE _WASI_EMULATED_MMAN)
    target_link_libraries(usearch_wasm PRIVATE c c++ c++abi wasi-emulated-mman)
endif()
set_target_properties(usearch_wasm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/wasm)
set_target_properties(usearch_wasm PROPERTIES CXX_STANDARD 11)
