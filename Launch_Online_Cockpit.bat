@echo off
setlocal
title D-SAAT Online Cockpit Launcher (Cloudflare HTTPS Tunnel)

cd /d "%~dp0"

echo ==============================================================
echo       D-SAAT Driver Safety AI — Online Deployment
echo ==============================================================
echo.

python tunnel.py

if %errorlevel% neq 0 (
    echo.
    echo An error occurred while running tunnel.py.
    pause
)
