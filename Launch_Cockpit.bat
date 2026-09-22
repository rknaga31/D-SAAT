@echo off
setlocal enabledelayedexpansion
title D-SAAT Localhost Cockpit Launcher

echo ==============================================================
echo   D-SAAT Driver Safety AI-Assisted (Offline Localhost)
echo ==============================================================

:: Change directory to script folder
cd /d "%~dp0"

:: Check if port 8000 is already listening
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] D-SAAT Cockpit server is already active on port 8000.
    echo Opening browser to http://localhost:8000 ...
    start http://localhost:8000
    goto end
)

echo [INFO] Starting D-SAAT Cockpit background service...
start "D-SAAT Cockpit Server" /min python web_app.py --no-browser

echo Waiting for Cockpit server to initialize...
set /a attempts=0

:check_loop
timeout /t 1 /nobreak >nul 2>&1
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Cockpit is LIVE on http://localhost:8000!
    echo Opening your browser now...
    start http://localhost:8000
    goto end
)

set /a attempts+=1
if !attempts! geq 12 (
    echo [NOTE] Launching browser directly...
    start http://localhost:8000
    goto end
)
goto check_loop

:end
timeout /t 2 /nobreak >nul 2>&1
exit /b 0
