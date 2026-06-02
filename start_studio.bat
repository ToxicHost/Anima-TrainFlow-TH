@echo off
setlocal
cd /d %~dp0

set "PY_EXE=%~dp0python_embeded\python.exe"
set "PYTHONPATH=%~dp0;%PYTHONPATH%"

if not exist "%PY_EXE%" (
    echo [ERROR] Portable Python not found at:
    echo "%PY_EXE%"
    pause
    exit
)

echo Starting Studio Trainer...
echo.

"%PY_EXE%" server.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server crashed. Check the error message above.
    pause
)
