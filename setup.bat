@echo off
title QwenVL Captioner Setup
color 0b
echo ===================================================
echo   QwenVL Image Captioner - Setup Environment
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10 or 3.11 from python.org
    pause
    exit /b 1
)

:: Create Virtual Environment
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv .venv
) else (
    echo [INFO] Virtual environment already exists.
)

:: Activate Virtual Environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

:: Upgrade PIP
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

:: Install Requirements
echo [INFO] Installing main dependencies...
pip install -r requirements.txt

:: Ask for Torch + CUDA
echo.
echo ===================================================
echo   OPTIONAL: Install Torch with CUDA 12.1 support?
echo   (Required if you plan to use GPU acceleration)
echo ===================================================
set /p install_torch="Install Torch+CUDA? (y/n): "
if /i "%install_torch%"=="y" (
    echo [INFO] Installing PyTorch with CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)

:: Ask for GGUF
echo.
echo ===================================================
echo   OPTIONAL: Install GGUF (llama-cpp-python) support?
echo   (Required if you plan to use GGUF models)
echo ===================================================
set /p install_gguf="Install llama-cpp-python? (y/n): "
if /i "%install_gguf%"=="y" (
    echo [INFO] Installing llama-cpp-python with CUDA support...
    set CMAKE_ARGS=-DGGML_CUDA=on
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
)

echo.
echo ===================================================
echo   Setup Complete!
echo   You can now launch the app using: run.bat
echo ===================================================
pause
