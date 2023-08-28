#!/bin/bash

check_task_status() {
  local task_name=$1
  local exit_code=$2

  if [ $exit_code -ne 0 ]; then
    echo "$task_name failed!"
    exit 1
  else
    echo "$task_name finished successfully!"
  fi
}

# Check vars for pack and push are set
required_vars=("PACKAGE_ID" "PACKAGE_VERSION" "NUGET_APIKEY" "NUGET_SERVER")
unset_vars=""

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    unset_vars="$unset_vars $var"
  fi
done

if [ -n "$unset_vars" ]; then
  echo "The following variables are not set:$unset_vars"
  exit 1
fi

# Get absolute path to the usearch dir
# FIX: realpath is not available on OSX, so use pwd
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR}/"

# Define project paths
REPO_ROOT=`cd -P "$SCRIPT_DIR../" && pwd`
REPO_ROOT="${REPO_ROOT}/"

# Clear package output folder
rm -rf ./packages/*

# Clean previous outputs
dotnet clean -c Release -p:RepoRoot="$REPO_ROOT"

# Build
dotnet build -c Release -p:Version=${PACKAGE_VERSION} -p:RepoRoot="$REPO_ROOT"

# Pack
dotnet pack -c Release -p:Version=${PACKAGE_VERSION} -p:RepoRoot="$REPO_ROOT"
EXIT_CODE=$?
check_task_status "Pack" $EXIT_CODE

# Push
dotnet nuget push "./packages/${PACKAGE_ID}.*.nupkg" -k ${NUGET_APIKEY} -s ${NUGET_SERVER}
EXIT_CODE=$?
check_task_status "NuGet Push" $EXIT_CODE
