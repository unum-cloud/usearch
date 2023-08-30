#!/bin/bash

# Get absolute path to the usearch dir
# FIX: realpath is not available on OSX, so use pwd
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR}/"

# Define project paths
REPO_ROOT=`cd -P "$SCRIPT_DIR../" && pwd`
REPO_ROOT="${REPO_ROOT}/"

# Clean and build the .NET project with RepoRoot property
dotnet clean -c Release -p:RepoRoot="$REPO_ROOT"
dotnet build -c Release -p:RepoRoot="$REPO_ROOT"
if [ $? -ne 0 ]; then
  echo "Build failed!"
  exit 1
fi

# Run the tests
dotnet test -c Release -p:RepoRoot="$REPO_ROOT" --no-build
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "Tests failed!"
  exit 1
else
  echo "Tests passed successfully!"
  exit 0
fi
