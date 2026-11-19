@echo off
setlocal
title NetraScan.AI - Portable Launcher

cd /d "%~dp0"

set "MPLBACKEND=Agg"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="

echo ==========================================================
echo                  NetraScan.AI
echo          Portable Cataract Classification
echo ==========================================================
echo.
echo Project: %CD%
echo.

if not exist "runtime\python\python.exe" (
    echo ERROR: Portable Python runtime not found.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERROR: NetraScan.AI application not found.
    pause
    exit /b 1
)

echo Starting NetraScan.AI...
echo.
echo Once the server is ready, open:
echo http://127.0.0.1:7860
echo.
echo Keep this window open while using the application.
echo To stop, use the Stop Server button in the interface.
echo.

"runtime\python\python.exe" -u "app.py"

echo.
echo NetraScan.AI server has stopped.
pause
endlocal
