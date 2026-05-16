@echo off
REM Records 10 seconds of audio to debug\test_capture.wav for device testing
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
python main.py --test-audio
pause
