@echo off
setlocal
cd /d "%~dp0"
echo Starting SignSpeak Backend...
echo Using Python Environment: venv310
"..\..\venv310\Scripts\python.exe" live_gesture_server.py
pause
