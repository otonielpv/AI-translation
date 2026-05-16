@echo off
REM AI-Translation - Church Live Translation Server

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python main.py %*
pause
