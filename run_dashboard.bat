@echo off
title D-SAAT Dashboard
echo ===================================================
echo   Starting D-SAAT Cockpit Telemetry Dashboard...
echo ===================================================
python -m streamlit run dashboard/app.py
pause
