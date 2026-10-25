@echo off
title QwenVL Captioner
color 0a

set "VENV_DIR=D:\Comfy\.venv\Scripts\activate"
set "DEACTIVATE_BAT=D:\Comfy\.venv\Scripts\deactivate.bat"

if not exist "%VENV_DIR%.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call %VENV_DIR%

echo [INFO] Starting QwenVL Image Captioner Web UI...
python app.py

if exist "%DEACTIVATE_BAT%" (
    echo [INFO] Deactivating virtual environment...
    call "%DEACTIVATE_BAT%"
)
