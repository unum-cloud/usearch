#!/bin/bash

# Get absolute path to the usearch csharp dir
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR}/"
CSHARP_DIR=`cd -P "${SCRIPT_DIR}.." && pwd`
CSHARP_DIR="${CSHARP_DIR}/"

rm -rf ${CSHARP_DIR}lib # Local temp path for native libs
rm -rf ${CSHARP_DIR}packages # Local temp path for nuget packages
find ${CSHARP_DIR} -type d -name 'bin' -exec rm -rf {} +
find ${CSHARP_DIR} -type d -name 'obj' -exec rm -rf {} +
