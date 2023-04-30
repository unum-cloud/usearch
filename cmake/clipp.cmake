include(FetchContent)
FetchContent_Declare(
    clipp
    GIT_REPOSITORY https://github.com/muellan/clipp
    GIT_TAG v1.2.3
)
FetchContent_MakeAvailable(clipp)
include_directories(${clipp_SOURCE_DIR}/include)
