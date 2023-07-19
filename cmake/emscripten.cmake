include(FetchContent)
FetchContent_Declare(
    emscripten
    GIT_REPOSITORY https://github.com/emscripten-core/emscripten
    GIT_TAG main
)
FetchContent_MakeAvailable(emscripten)
include_directories(${emscripten_SOURCE_DIR}/include)
