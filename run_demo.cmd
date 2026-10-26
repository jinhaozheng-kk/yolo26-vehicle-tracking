@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please follow README.md to set up the environment first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" track.py --show %*
pause
