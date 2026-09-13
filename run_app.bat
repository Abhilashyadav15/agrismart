@echo off
title AgriSmart Web Application
cd /d "%~dp0"

echo ========================================================
echo Starting AgriSmart inside isolated .venv...
echo ========================================================

rem Launch default browser after 2 seconds in background
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8501"

rem Run Streamlit using the project's dedicated .venv Python
".venv\Scripts\python.exe" -m streamlit run app.py

pause
