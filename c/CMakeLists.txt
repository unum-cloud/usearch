set(USEARCH_PUNNED_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../fp16/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../robin-map/include"
)

add_executable(usearch lib.cpp)
target_include_directories(usearch PRIVATE ${USEARCH_PUNNED_INCLUDE_DIRS})
set_target_properties(usearch PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c)
set_target_properties(usearch PROPERTIES CXX_STANDARD 11)
