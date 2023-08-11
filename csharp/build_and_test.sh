#!/bin/bash

# Get absolute path to the usearch dir
# FIX: realpath is not available on OSX, so use pwd
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Define project paths
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
C_DIR="$PROJECT_ROOT/c"
CSHARP_DIR="$PROJECT_ROOT/csharp"
TEST_PROJ_PATH="$CSHARP_DIR/src/LibUSearch.Tests"
TEST_PROJ_RUNTIMES_TEMP_PATH="$TEST_PROJ_PATH/runtimes"

# Determine the OS type (linux or macos) and set the shared library extension accordingly
case "$(uname -s)" in
    Darwin)
        SHARED_LIB_EXT="dylib"
        ;;
    Linux)
        SHARED_LIB_EXT="so"
        ;;
    *)
        echo "Unsupported operating system"
        exit 1
        ;;
esac

# Build the native dynamic lib
cd "$C_DIR" || { echo "C directory not found in $C_DIR"; exit 1; }
if [ ! -f Makefile ]; then
  echo "Makefile not found in $C_DIR"
  exit 1
fi
make libusearch_c.$SHARED_LIB_EXT
if [ $? -ne 0 ]; then
  echo "Build failed!"
  exit 1
fi

# Switch to the C# directory
cd "$CSHARP_DIR" || { echo "C# directory not found in $CSHARP_DIR"; exit 1; }

# Clean and build the .NET project
dotnet clean
dotnet build --configuration Release
if [ $? -ne 0 ]; then
  echo "Build failed!"
  exit 1
fi

# Create a temporary path for dynamic library and move it there
mkdir -p "$TEST_PROJ_RUNTIMES_TEMP_PATH"
mv "$C_DIR/libusearch_c.$SHARED_LIB_EXT" "$TEST_PROJ_RUNTIMES_TEMP_PATH"

# Run the tests
dotnet test --configuration Release
EXIT_CODE=$?

# Clear the temporary path
rm -rf "$TEST_PROJ_RUNTIMES_TEMP_PATH"

if [ $EXIT_CODE -ne 0 ]; then
  echo "Tests failed!"
  exit 1
else
  echo "Tests passed successfully!"
  exit 0
fi
