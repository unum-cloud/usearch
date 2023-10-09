include(emscripten)

# Setting the compiler globally is never a good idea! set(CMAKE_C_COMPILER emcc) set(CMAKE_CXX_COMPILER em++)

add_executable(usearch_wasm lib.cpp)
setup_target(usearch_wasm)
set_target_properties(usearch_wasm PROPERTIES CXX_STANDARD 11)
