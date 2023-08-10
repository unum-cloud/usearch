#!/bin/bash

dotnet restore
dotnet clean
dotnet build --configuration Release

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

dotnet test --configuration Release

if [ $? -ne 0 ]; then
    echo "Tests failed!"
    exit 1
else
    echo "Tests passed successfully!"
fi
