@echo off
title QwenVL Batch Captioner
color 0a

set "VENV_DIR=%~dp0.venv\Scripts\activate.bat"

if not exist "%VENV_DIR%" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call "%VENV_DIR%"

echo [INFO] Starting batch captioning...
python "%~dp0batch_caption.py" %*

pause
