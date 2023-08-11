@echo off
setlocal EnableDelayedExpansion

:: Get the absolute path of the directory where the script is located
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

:: Set up project paths
set PROJECT_ROOT=%SCRIPT_DIR%\..
set C_DIR=%PROJECT_ROOT%\c
set CSHARP_DIR=%PROJECT_ROOT%\csharp
set TEST_PROJ_PATH=%CSHARP_DIR%\src\LibUSearch.Tests
set TEST_PROJ_RUNTIMES_TEMP_PATH=%TEST_PROJ_PATH%\runtimes

:: Build the native DLL
cd %C_DIR%
if not exist makefile (
    echo Makefile not found in %C_DIR%
    exit /b 1
)
make libusearch_c.dll
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

:: Switch to the C# directory
cd %CSHARP_DIR%

:: Clean and build the .NET project
dotnet clean
dotnet build --configuration Release
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

:: Create a temporary path for the DLL and move it there
mkdir "%TEST_PROJ_RUNTIMES_TEMP_PATH%"
move "%C_DIR%\libusearch_c.dll" "%TEST_PROJ_RUNTIMES_TEMP_PATH%"

:: Run the tests
dotnet test --configuration Release
set EXIT_CODE=%ERRORLEVEL%

:: Clear the temporary path
rmdir /s /q "%TEST_PROJ_RUNTIMES_TEMP_PATH%"

if %EXIT_CODE% neq 0 (
    echo Tests failed!
    exit /b 1
) else (
    echo Tests passed successfully!
    exit /b 0
)