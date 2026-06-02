@echo off
setlocal
cd /d %~dp0

if not exist "python_embeded\python.exe" (
    echo [ERROR] python_embeded folder not found!
    pause
    exit
)

set "PY_EXE=%~dp0python_embeded\python.exe"

echo [1/4] Installing PyTorch (CUDA 12.8)...
"%PY_EXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo [2/4] Installing Dependencies from sd-scripts...
cd /d "%~dp0training\sd-scripts"
"%PY_EXE%" -m pip install -r requirements.txt

echo [3/4] Linking sd-scripts library...
"%PY_EXE%" -m pip install -e .

cd /d %~dp0

echo [4/4] Installing Studio Trainer server deps (FastAPI + Uvicorn)...
"%PY_EXE%" -m pip install fastapi uvicorn

echo.
echo Installation Complete!
echo Run the new UI via start_studio.bat (or the legacy Gradio UI via start_trainer.bat)
pause