@echo off

rem Set the version number.
set version=v2.1.0

echo Get library version!
timeout /t 2
rem Split the first 'v' from the version number.
for /f "tokens=1 delims=v" %%a in ('echo %version%') do (
    set first_token=%%a
)

rem Save the first token in the lib_version variable.
set lib_version=%first_token%
echo Getting library version is complete!

echo Download usearch archive
rem Download the file.
timeout /t 2
curl -LO https://github.com/unum-cloud/usearch/releases/download/%version%/usearch_windows_%lib_version%.tar
echo Download is complete!

echo Extract archive!
rem Extract archive.
timeout /t 1
tar -xf usearch_windows_%lib_version%.tar
echo Extract is complete!

rem Remove the archive.
del usearch_windows_%lib_version%.tar

echo Extraction complete and downloaded .tar file removed.
timeout /t 1
exit /b