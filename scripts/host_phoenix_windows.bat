@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem Phoenix Windows seminar host script
rem
rem Usage:
rem   host_phoenix_windows.bat <repo-url> [install-dir] [port] [branch]
rem   host_phoenix_windows.bat
rem
rem Example:
rem   host_phoenix_windows.bat https://github.com/YOUR_USER/YOUR_REPO.git C:\PhoenixHost 8501 phoenix-refactor
rem
rem If this script is run from inside an existing Phoenix Git checkout, repo-url
rem and branch are detected from that checkout's origin/current branch.
rem
rem What it does:
rem   1. Clones or updates the Phoenix repo.
rem   2. Creates/updates a "phoenix" conda environment from cellbench/environment.yml
rem      when conda is available.
rem   3. Falls back to a local Python venv when conda is not available.
rem   4. Offers to open the Windows Firewall port when run as Administrator.
rem   5. Starts Streamlit on 0.0.0.0 so other machines can connect.
rem ---------------------------------------------------------------------------

set "REPO_URL=%~1"
set "INSTALL_DIR=%~2"
set "PORT=%~3"
set "BRANCH=%~4"
set "ENV_NAME=phoenix"
set "SERVER_ADDRESS=0.0.0.0"
set "FIREWALL_RULE=Phoenix Streamlit %PORT%"
set "SCRIPT_DIR=%~dp0"
set "SOURCE_REPO=%SCRIPT_DIR%.."
set "REMOTE_NAME="

if "%PORT%"=="" set "PORT=8501"
if "%INSTALL_DIR%"=="" set "INSTALL_DIR=%USERPROFILE%\PhoenixHost"

if "%REPO_URL%"=="" (
    if exist "%SOURCE_REPO%\.git" (
        for /f "usebackq delims=" %%G in (`git -C "%SOURCE_REPO%" config --get remote.origin.url 2^>nul`) do set "REPO_URL=%%G"
        if "!REPO_URL!"=="" (
            for /f "usebackq delims=" %%R in (`git -C "%SOURCE_REPO%" remote 2^>nul`) do (
                if "!REMOTE_NAME!"=="" set "REMOTE_NAME=%%R"
            )
            if not "!REMOTE_NAME!"=="" (
                for /f "usebackq delims=" %%G in (`git -C "%SOURCE_REPO%" config --get remote.!REMOTE_NAME!.url 2^>nul`) do set "REPO_URL=%%G"
            )
        )
        if not "!REPO_URL!"=="" (
            echo Detected Git remote URL from current checkout:
            echo   !REPO_URL!
        )
        if "%BRANCH%"=="" (
            for /f "usebackq delims=" %%B in (`git -C "%SOURCE_REPO%" branch --show-current 2^>nul`) do set "BRANCH=%%B"
            if not "!BRANCH!"=="" (
                echo Detected current branch from checkout:
                echo   !BRANCH!
            )
        )
    )
)

if "%REPO_URL%"=="" (
    echo.
    echo No repository URL was provided.
    echo I also could not detect remote.origin.url from this script's checkout.
    echo.
    echo To enable automatic detection next time, configure a remote in your
    echo local Phoenix repo, for example:
    echo.
    echo   git remote add origin https://gitlab.example.com/GROUP/PROJECT.git
    echo.
    set /p "REPO_URL=Paste the Phoenix Git repository URL: "
)

if "%REPO_URL%"=="" (
    echo ERROR: repository URL is required.
    exit /b 1
)

set "APP_DIR=%INSTALL_DIR%\Phoenix"
set "FIREWALL_RULE=Phoenix Streamlit %PORT%"

echo.
echo ============================================================
echo Phoenix seminar host setup
echo ============================================================
echo Repo:        %REPO_URL%
echo Install dir: %APP_DIR%
echo Port:        %PORT%
if not "%BRANCH%"=="" echo Branch:      %BRANCH%
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git was not found on PATH.
    echo Install Git for Windows first: https://git-scm.com/download/win
    exit /b 1
)

if exist "%APP_DIR%\.git" (
    echo Existing Phoenix checkout found.
    if not "%BRANCH%"=="" (
        echo Checking out branch "%BRANCH%"...
        git -C "%APP_DIR%" fetch origin "%BRANCH%"
        if errorlevel 1 (
            echo ERROR: could not fetch branch "%BRANCH%".
            exit /b 1
        )
        git -C "%APP_DIR%" checkout "%BRANCH%"
        if errorlevel 1 (
            echo ERROR: could not check out branch "%BRANCH%".
            exit /b 1
        )
        echo Updating with git pull --ff-only origin "%BRANCH%"...
        git -C "%APP_DIR%" pull --ff-only origin "%BRANCH%"
    ) else (
        echo Updating with git pull --ff-only...
        git -C "%APP_DIR%" pull --ff-only
    )
    if errorlevel 1 (
        echo ERROR: Could not update existing checkout.
        echo You can delete "%APP_DIR%" and rerun this script if needed.
        exit /b 1
    )
) else (
    if exist "%APP_DIR%" (
        echo ERROR: "%APP_DIR%" exists but is not a Git repository.
        echo Move or delete that folder, then rerun this script.
        exit /b 1
    )
    if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
    echo Cloning Phoenix...
    if not "%BRANCH%"=="" (
        git clone --branch "%BRANCH%" --single-branch "%REPO_URL%" "%APP_DIR%"
    ) else (
        git clone "%REPO_URL%" "%APP_DIR%"
    )
    if errorlevel 1 (
        echo ERROR: git clone failed.
        exit /b 1
    )
)

cd /d "%APP_DIR%"
if errorlevel 1 (
    echo ERROR: could not enter "%APP_DIR%".
    exit /b 1
)

echo.
echo ============================================================
echo Firewall
echo ============================================================
net session >nul 2>&1
if errorlevel 1 (
    echo This script is not running as Administrator.
    echo.
    echo If other people cannot open Phoenix, rerun this script as Administrator
    echo or ask IT to allow inbound TCP port %PORT% on this workstation.
    echo.
    echo Administrator command:
    echo netsh advfirewall firewall add rule name="%FIREWALL_RULE%" dir=in action=allow protocol=TCP localport=%PORT% profile=any
) else (
    echo Administrator privileges detected.
    choice /C YN /N /M "Open Windows Firewall TCP port %PORT% for all network profiles? [Y/N] "
    if errorlevel 2 (
        echo Firewall rule not changed.
    ) else (
        netsh advfirewall firewall delete rule name="%FIREWALL_RULE%" >nul 2>&1
        netsh advfirewall firewall add rule name="%FIREWALL_RULE%" dir=in action=allow protocol=TCP localport=%PORT% profile=any
        if errorlevel 1 (
            echo WARNING: Could not add firewall rule.
        ) else (
            echo Firewall rule added: %FIREWALL_RULE%
        )
    )
)

echo.
echo ============================================================
echo Python environment
echo ============================================================

where conda >nul 2>&1
if not errorlevel 1 (
    if exist "cellbench\environment.yml" (
        echo Conda found. Creating or updating conda environment "%ENV_NAME%"...
        call conda env list | findstr /R /C:"^%ENV_NAME%[ ]" >nul 2>&1
        if errorlevel 1 (
            call conda env create -f cellbench\environment.yml
        ) else (
            call conda env update -n "%ENV_NAME%" -f cellbench\environment.yml --prune
        )
        if errorlevel 1 (
            echo ERROR: conda environment setup failed.
            exit /b 1
        )
        set "RUN_CMD=conda run -n %ENV_NAME% python -m streamlit run phoenix/app.py --server.address=%SERVER_ADDRESS% --server.port=%PORT% --server.headless=true"
        goto run_app
    )
)

echo Conda was not found, or cellbench\environment.yml is missing.
echo Falling back to a local Python virtual environment.

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_LAUNCHER=py -3"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python was not found on PATH.
        exit /b 1
    )
    set "PYTHON_LAUNCHER=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv...
    %PYTHON_LAUNCHER% -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create virtual environment.
        exit /b 1
    )
)

call .venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1

call .venv\Scripts\python.exe -m pip install numpy pandas scipy matplotlib "streamlit>=1.58" "pybamm==26.6.2.0"
if errorlevel 1 (
    echo ERROR: pip dependency installation failed.
    exit /b 1
)

set "RUN_CMD=.venv\Scripts\python.exe -m streamlit run phoenix/app.py --server.address=%SERVER_ADDRESS% --server.port=%PORT% --server.headless=true"

:run_app
set "STREAMLIT_SERVER_ADDRESS=%SERVER_ADDRESS%"
set "STREAMLIT_SERVER_PORT=%PORT%"
set "STREAMLIT_SERVER_HEADLESS=true"

echo.
echo ============================================================
echo Starting Phoenix
echo ============================================================
echo.
echo Share one of these addresses with seminar participants:
echo.
echo   http://WORKSTATION-IP:%PORT%
echo.
echo Candidate IPv4 addresses on this machine:
ipconfig | findstr /R /C:"IPv4.*:"
echo.
echo If participants cannot connect:
echo   1. Check that they are on the same network or VPN.
echo   2. Check Windows Firewall / IT firewall rules.
echo   3. Try opening http://localhost:%PORT% on this workstation first.
echo.
echo After startup, this command should show 0.0.0.0:%PORT%, not 127.0.0.1:%PORT%:
echo   netstat -ano ^| findstr :%PORT%
echo.
echo Running:
echo %RUN_CMD%
echo.

%RUN_CMD%

endlocal
