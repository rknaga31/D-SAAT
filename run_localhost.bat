@echo off
setlocal enabledelayedexpansion
title D-SAAT Localhost Cockpit

echo ==============================================================
echo   D-SAAT Localhost Web Cockpit (Offline Ready)
echo   Dashboard URL: http://localhost:8000
echo ==============================================================

cd /d "%~dp0"

:: Check if port 8000 is already running
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] D-SAAT Cockpit is already running on port 8000.
    echo Opening browser to http://localhost:8000 ...
    start http://localhost:8000
    timeout /t 3 >nul
    exit /b 0
)

python web_app.py
pause
