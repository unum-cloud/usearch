@echo off
setlocal EnableDelayedExpansion

:: Get the absolute paths of the directory where the script is located and repo root
set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\\

:: Switch to the C# directory
cd %REPO_ROOT%csharp\

:: Clean and build the .NET project
dotnet clean -c Release -p:RepoRoot="%REPO_ROOT%"
dotnet build -c Release -p:RepoRoot="%REPO_ROOT%"
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

:: Run the tests
dotnet test -c Release -p:RepoRoot="%REPO_ROOT%" --no-build
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% neq 0 (
    echo Tests failed!
    exit /b 1
) else (
    echo Tests passed successfully!
    exit /b 0
)

endlocal
