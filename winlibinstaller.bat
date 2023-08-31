@echo off

rem Set usearch release version.
set version=v2.0.2

rem Get Windows architecture
for /f "tokens=2 delims= " %%a in ('wmic OS get OSArchitecture ^| find "64"') do set "arch=x64"
if "%arch%"=="" set "arch=x86"

echo Split library version from release version!
timeout /t 1
rem Split the first 'v' from the version number.
for /f "tokens=1 delims=v" %%a in ('echo %version%') do (
    set lib_version=%%a
)

echo Download usearch library archive!
rem Download the file.
timeout /t 2
curl -LO https://github.com/gurgenyegoryan/usearch/releases/download/%version%/usearch_windows_%arch%_%lib_version%.tar
echo Download is complete!

echo Extract archive!
rem Extract archive.
timeout /t 1
tar -xf usearch_windows_%arch%_%lib_version%.tar
echo Extract is complete!

rem Remove the archive.
del usearch_windows_%arch%_%lib_version%.tar

echo Extraction complete and downloaded .tar file removed.
timeout /t 1
exit /b