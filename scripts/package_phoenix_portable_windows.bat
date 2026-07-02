@echo off
setlocal EnableExtensions

rem ---------------------------------------------------------------------------
rem Build a portable Phoenix Windows folder/zip.
rem
rem Run this on Windows from the Phoenix repo:
rem   scripts\package_phoenix_portable_windows.bat
rem
rem Output:
rem   dist\PhoenixPortableWindows\
rem   dist\PhoenixPortableWindows.zip
rem
rem This is intentionally not a single-file .exe. It is a portable app folder
rem with one launcher, which is much more reliable for Streamlit + PyBaMM.
rem ---------------------------------------------------------------------------

set "ROOT_DIR=%~dp0.."
for %%I in ("%ROOT_DIR%") do set "ROOT_DIR=%%~fI"

set "BUILD_DIR=%ROOT_DIR%\build\portable_windows"
set "DIST_ROOT=%ROOT_DIR%\dist"
set "DIST_DIR=%DIST_ROOT%\PhoenixPortableWindows"
set "BUILD_ENV=%BUILD_DIR%\env"
set "ENV_ARCHIVE=%BUILD_DIR%\phoenix-env-windows.tar.gz"
set "ZIP_FILE=%DIST_ROOT%\PhoenixPortableWindows.zip"

echo.
echo ============================================================
echo Building Phoenix portable Windows package
echo ============================================================
echo Repo:   %ROOT_DIR%
echo Output: %DIST_DIR%
echo.

if "%ROOT_DIR:~0,2%"=="\\" (
    echo ERROR: this script is running from a UNC/network path:
    echo   %ROOT_DIR%
    echo.
    echo This often happens when launching the script from WSL, for example
    echo \\wsl.localhost\...\Phoenix. Conda environments and portable Windows
    echo packages are not reliable from that location.
    echo.
    echo Please clone Phoenix to a normal Windows path first, for example:
    echo.
    echo   C:\PhoenixBuild\Phoenix
    echo.
    echo Then open Anaconda Prompt and run:
    echo.
    echo   cd C:\PhoenixBuild\Phoenix
    echo   scripts\package_phoenix_portable_windows.bat
    echo.
    pause
    exit /b 1
)

where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: conda was not found on PATH.
    echo Install Miniconda/Mambaforge or run this from Anaconda Prompt.
    pause
    exit /b 1
)

if not exist "%ROOT_DIR%\cellbench\environment.yml" (
    echo ERROR: missing %ROOT_DIR%\cellbench\environment.yml
    pause
    exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%DIST_ROOT%" mkdir "%DIST_ROOT%"

set "ENV_SETUP_FAILED=0"
if exist "%BUILD_ENV%\conda-meta" (
    echo Updating build environment...
    call conda env update -p "%BUILD_ENV%" -f "%ROOT_DIR%\cellbench\environment.yml" --prune
) else (
    echo Creating build environment...
    call conda env create -p "%BUILD_ENV%" -f "%ROOT_DIR%\cellbench\environment.yml"
)
if errorlevel 1 (
    set "ENV_SETUP_FAILED=1"
)

if "%ENV_SETUP_FAILED%"=="1" (
    echo.
    echo WARNING: environment creation/update failed.
    echo This is often caused by a corrupted conda package cache, for example:
    echo   InvalidArchiveError^(... .conda^)
    echo.
    echo Cleaning conda package caches and retrying once...
    call conda clean --all -y
    if exist "%BUILD_ENV%" rmdir /s /q "%BUILD_ENV%"
    call conda env create -p "%BUILD_ENV%" -f "%ROOT_DIR%\cellbench\environment.yml"
    if errorlevel 1 (
        echo.
        echo ERROR: environment creation/update failed even after cache cleanup.
        echo.
        echo Manual fix to try in Anaconda Prompt:
        echo   conda clean --all -y
        echo   rmdir /s /q "%BUILD_ENV%"
        echo   scripts\package_phoenix_portable_windows.bat
        echo.
        echo If the error names one package archive, delete that broken file
        echo from your Anaconda pkgs cache and rerun the script.
        pause
        exit /b 1
    )
)

echo Ensuring conda-pack is available...
call conda install -n base -c conda-forge conda-pack -y
if errorlevel 1 (
    echo ERROR: could not install conda-pack into base.
    pause
    exit /b 1
)

echo Packing environment...
if exist "%ENV_ARCHIVE%" del "%ENV_ARCHIVE%"
call conda run -n base conda-pack -p "%BUILD_ENV%" -o "%ENV_ARCHIVE%" --force
if errorlevel 1 (
    echo ERROR: conda-pack failed.
    pause
    exit /b 1
)

echo Recreating portable folder...
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
mkdir "%DIST_DIR%"
mkdir "%DIST_DIR%\app"
mkdir "%DIST_DIR%\env"

echo Extracting portable environment...
tar -xzf "%ENV_ARCHIVE%" -C "%DIST_DIR%\env"
if errorlevel 1 (
    echo ERROR: could not extract environment archive.
    pause
    exit /b 1
)

echo Copying Phoenix source tree...
robocopy "%ROOT_DIR%" "%DIST_DIR%\app" /E ^
    /XD ".git" ".venv" "__pycache__" "build" "dist" ".pytest_cache" ".mypy_cache" ^
    /XF "*.pyc" "*.pyo" ".DS_Store"
if %ERRORLEVEL% GEQ 8 (
    echo ERROR: robocopy failed.
    pause
    exit /b 1
)

copy "%ROOT_DIR%\scripts\launch_phoenix_local_windows.bat" "%DIST_DIR%\Phoenix.bat" >nul
if errorlevel 1 (
    echo ERROR: could not copy launcher.
    pause
    exit /b 1
)

echo Writing participant README...
(
    echo Phoenix portable Windows package
    echo ================================
    echo.
    echo How to run:
    echo 1. Extract PhoenixPortableWindows.zip.
    echo 2. Open the extracted folder.
    echo 3. Double-click Phoenix.bat.
    echo 4. Your browser should open http://127.0.0.1:8501.
    echo.
    echo How to stop:
    echo - Close the Phoenix command window or press Ctrl+C in it.
    echo.
    echo Notes:
    echo - You do not need Git.
    echo - You do not need to install Python.
    echo - You do not need to install conda.
    echo - This starts Phoenix only on your own computer.
    echo - The first launch may take a bit longer while the bundled environment is prepared.
) > "%DIST_DIR%\README_FIRST.txt"

echo Creating zip archive...
if exist "%ZIP_FILE%" del "%ZIP_FILE%"
powershell -NoProfile -Command "Compress-Archive -Path '%DIST_DIR%\*' -DestinationPath '%ZIP_FILE%' -Force"
if errorlevel 1 (
    echo PowerShell Compress-Archive failed. Trying Windows tar fallback...
    if exist "%ZIP_FILE%" del "%ZIP_FILE%"
    tar -a -c -f "%ZIP_FILE%" -C "%DIST_DIR%" .
)

if exist "%ZIP_FILE%" (
    echo Zip created:
    echo   %ZIP_FILE%
) else (
    echo.
    echo WARNING: zip archive creation failed, but folder output exists:
    echo   %DIST_DIR%
    echo.
    echo You can manually zip the contents of that folder, or give people the
    echo whole PhoenixPortableWindows folder.
)

echo.
echo Done.
echo.
echo Test locally:
echo   %DIST_DIR%\Phoenix.bat
echo.
echo Distribution:
echo   Give people PhoenixPortableWindows.zip, have them extract it, then run Phoenix.bat.
echo.
echo Output folder:
echo   %DIST_ROOT%
echo.

if exist "%ZIP_FILE%" (
    explorer /select,"%ZIP_FILE%"
) else (
    explorer "%DIST_DIR%"
)

pause

endlocal
