#!/bin/bash

# Get absolute path to the usearch dir
# FIX: realpath is not available on OSX, so use pwd
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR}/"

# Define project paths
REPO_ROOT=`cd -P "${SCRIPT_DIR}../.." && pwd`
REPO_ROOT="${REPO_ROOT}/"
C_DIR="${REPO_ROOT}c/"
CSHARP_DIR="${REPO_ROOT}csharp/"
BUILD_ARTIFACTS_DIR="${CSHARP_DIR}lib/"

LIB_FILE="libusearch_c"

# Determine the architecture type (only x64 is supported)
ARCH_TYPE="$(uname -m)"
case "$ARCH_TYPE" in
    x86_64)
        ARCH="x64"
        ;;
    # i386|i486|i586|i686)
    #     ARCH="x86"
    #     ;;
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

MAKE_TARGET="${LIB_FILE}"

# Build the native dynamic lib
cd "$C_DIR" || { echo "C directory not found in $C_DIR"; exit 1; }
if [ ! -f Makefile ]; then
  echo "Makefile not found in $C_DIR"
  exit 1
fi
make "${MAKE_TARGET}"
if [ $? -ne 0 ]; then
  echo "Build failed!"
  exit 1
fi

# Create output dir for shared lib and copy it there
mkdir -p "$BUILD_ARTIFACTS_DIR"
cp "${C_DIR}/${LIB_FILE}" "$BUILD_ARTIFACTS_DIR"
