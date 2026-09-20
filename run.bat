@echo off
title QwenVL Captioner
color 0a

set "VENV_DIR=%~dp0.venv\Scripts\activate.bat"

if not exist "%VENV_DIR%" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call "%VENV_DIR%"

echo [INFO] Starting QwenVL Image Captioner Web UI...
python "%~dp0app.py" --llm-dir llm

pause
