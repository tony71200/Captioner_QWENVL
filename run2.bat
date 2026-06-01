@echo off
title QwenVL Captioner
color 0a

VENV_DIR = ".venv\Scripts\activate"


echo [INFO] Activating virtual environment...
call %VENV_DIR%

echo [INFO] Starting QwenVL Image Captioner Web UI...
python app.py --llm-dir llm

pause
