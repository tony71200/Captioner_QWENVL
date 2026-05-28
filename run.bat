@echo off
title QwenVL Captioner
color 0a

VENV_DIR = ".venv\Scripts\activate"

if not exist %VENV_DIR% + ".bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call %VENV_DIR%

echo [INFO] Starting QwenVL Image Captioner Web UI...
python app.py --llm-dir llm

pause
