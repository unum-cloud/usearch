#!/bin/bash

# Get absolute path to the usearch dir
# FIX: realpath is not available on OSX, so use pwd
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR}/"

# Define project paths
REPO_ROOT=`cd -P "${SCRIPT_DIR}../.." && pwd`
REPO_ROOT="${REPO_ROOT}/"
CSHARP_DIR="${REPO_ROOT}csharp/"
BUILD_ARTIFACTS_DIR="${CSHARP_DIR}lib/"

LIB_FILE="libusearch_c"

# Determine the architecture type (only x64 is supported)
ARCH_TYPE="$(uname -m)"
case "$ARCH_TYPE" in
    x86_64)
        ARCH="x64"
        ;;
    *)
        echo "Unsupported architecture"
        exit 1
        ;;
esac
echo "Detected architecture: $ARCH"

# Determine the OS type (linux or macos) and set the shared library extension accordingly
case "$(uname -s)" in
    Darwin)
        BUILD_ARTIFACTS_DIR+="runtimes/osx-${ARCH}/native/"
        LIB_FILE+=".dylib"
        ;;
    Linux)
        BUILD_ARTIFACTS_DIR+="runtimes/linux-${ARCH}/native/"
        LIB_FILE+=".so"
        ;;
    *)
        echo "Unsupported operating system"
        exit 1
        ;;
esac

# Compile with CMake
cmake -B build_release -DCMAKE_BUILD_TYPE=Release -DUSEARCH_BUILD_TEST_CPP=0 -DUSEARCH_BUILD_LIB_C=1 -DUSEARCH_USE_OPENMP=0 -DUSEARCH_USE_NATIVE_F16=0 -DUSEARCH_USE_SIMSIMD=1
cmake --build build_release --config Release
if [ $? -ne 0 ]; then
  echo "Build failed!"
  exit 1
fi

# Create output dir for shared lib and copy it there
mkdir -p "$BUILD_ARTIFACTS_DIR"
cp "${REPO_ROOT}/build_release/${LIB_FILE}" "$BUILD_ARTIFACTS_DIR"
