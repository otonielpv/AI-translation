@echo off
REM Lists available audio input devices
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
python main.py --list-devices
pause
