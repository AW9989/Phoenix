@echo off
setlocal EnableExtensions

rem ---------------------------------------------------------------------------
rem Phoenix portable Windows launcher
rem
rem Expected portable folder layout:
rem   PhoenixPortableWindows\
rem     Phoenix.bat
rem     app\
rem       phoenix\app.py
rem       Parameter_Sets\...
rem     env\
rem       python.exe
rem ---------------------------------------------------------------------------

set "ROOT_DIR=%~dp0"
set "APP_DIR=%ROOT_DIR%app"
set "ENV_DIR=%ROOT_DIR%env"
set "PORT=%PHOENIX_PORT%"

if "%PORT%"=="" set "PORT=8501"

if not exist "%APP_DIR%\phoenix\app.py" (
    echo ERROR: Could not find Phoenix app at:
    echo   %APP_DIR%\phoenix\app.py
    echo.
    echo This launcher is meant to be run from the portable Phoenix folder.
    pause
    exit /b 1
)

if not exist "%ENV_DIR%\python.exe" (
    echo ERROR: Could not find bundled Python at:
    echo   %ENV_DIR%\python.exe
    pause
    exit /b 1
)

if exist "%ENV_DIR%\Scripts\conda-unpack.exe" (
    if not exist "%ENV_DIR%\.phoenix_unpacked" (
        echo Preparing portable Python environment for this machine...
        "%ENV_DIR%\Scripts\conda-unpack.exe"
        if errorlevel 1 (
            echo ERROR: conda-unpack failed.
            pause
            exit /b 1
        )
        echo ok>"%ENV_DIR%\.phoenix_unpacked"
    )
)

set "STREAMLIT_SERVER_ADDRESS=127.0.0.1"
set "STREAMLIT_SERVER_PORT=%PORT%"
set "STREAMLIT_SERVER_HEADLESS=true"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

echo.
echo Starting Phoenix locally...
echo.
echo Browser URL:
echo   http://127.0.0.1:%PORT%
echo.
echo This local launcher does not expose Phoenix to the network.
echo Close this window or press Ctrl+C to stop Phoenix.
echo.

powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:%PORT%'" >nul 2>&1

"%ENV_DIR%\python.exe" -m streamlit run "%APP_DIR%\phoenix\app.py" --server.address=127.0.0.1 --server.port=%PORT% --server.headless=true --browser.gatherUsageStats=false

endlocal
